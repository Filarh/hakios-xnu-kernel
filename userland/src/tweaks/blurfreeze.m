/*
 * blurfreeze.dylib: a SpringBoard tweak that, while a material's blur animates (the Control Center opening or closing),
 * gives its backdrop a STATIC source -- a snapshot of what is behind it, taken once -- instead of the live screen.
 * Prototype for an A/B on the phone; nothing changes unless /var/tmp/blurfreeze.args asks for it.
 *
 * Why (dev-notes/investigacion/22 §1.2, 25 §2.7): each frame of the animation CoreAnimation redraws the whole screen
 * under the material (the main pass, ~9-10 ms of GPU), captures it at full resolution (~4 ms), reduces it (~2.2 ms) and
 * only then blurs it with the radius of that frame. The content under the material does not change during the
 * animation; only the radius (and the material's colour filters and zoom) does.
 *
 *   mode=1  "source": an opaque layer with the snapshot goes right under the material, which keeps its live backdrop and
 *           its live filters. What the backdrop captures is the same picture, so the look is the same (exact at scale=1);
 *           what can go is the redraw of the layers under the snapshot, if CoreAnimation culls layers hidden behind an
 *           opaque one (QuartzCore has Update::add_opaque_shape, Updater::union_opaque_rect and
 *           layer_may_be_hidden_culled: unverified that it applies here). The capture and the reduction stay.
 *   mode=2  "filter": the same snapshot layer, but the material's backdrop is disabled for the session and its filters
 *           (blur, colour matrices, curves, opacity matrix) and their animations are mirrored onto the snapshot layer,
 *           so the blur of each frame is taken from the static snapshot (at scale=S of the screen) and no capture of the
 *           screen is made at all. Apple's own MTMaterialLayer -setContentReplacedWithSnapshot: does half of this (it
 *           disables the backdrop and puts a render-server snapshot in its contents, but sets the filters to nil).
 *           Differences in the look are possible: edges (inputNormalizeEdges), the zoom (approximated by a scale), and
 *           the resolution of the source at small radii.
 *
 * A session starts on the first weighting change (or weighting/filter animation) of a material whose recipe is in the
 * allow list, lasts while changes keep coming (slack) and until the animations end, and never more than maxms; then the
 * live backdrop is given back, with any filter animation still running. The snapshot excludes the material's own
 * CAContext (its window: the Control Center's) and every SpringBoard window at or above its level (the status bar).
 *
 * /var/tmp/blurfreeze.args (read when no session is running, at most once a second), words:
 *   mode=0|1|2        0 (default, also with no file): nothing
 *   scale=F           the snapshot's resolution relative to the screen (default 1 for mode 1, 0.5 for mode 2)
 *   maxms=N           the longest a session lasts (default 1500)
 *   slackms=N         how long after the last change a session stays (default 250)
 *   recipes=a,b       the materials it applies to, * for all (default modulesBackground,homeScreenOverlay: the Control
 *                     Center's background, and the overlay of the Today panel and the App Library)
 *   dump=1            each snapshot is also written to /var/tmp/blurfreeze-N.raw (BGRA, its size in the log line), and
 *                     every animation added to a material and its weighting changes are logged (a weighting line per
 *                     material at most every 100 ms; nothing is formatted without dump=1)
 *   quiet=1           no log lines but the errors
 *
 * Crash guard: /var/tmp/blurfreeze.armed exists while a session runs (it is written before the snapshot and removed when
 * the session ends). A SpringBoard that starts and finds it died in a session: the file becomes /var/tmp/blurfreeze.tripped
 * (with the count of trips) and the tweak installs nothing. While blurfreeze.tripped exists it stays off; rm it to re-arm.
 *
 * Each session says on the console: "blurfreeze: session N mode M recipe R rect ... snapshot W x H in X ms, excluded
 * contexts [...]" and at the end "blurfreeze: session N ended after X ms (changes C, animations A)".
 *
 * Built by blurfreeze.sh (xcrun clang against the iPhoneOS SDK, iOS 16 minimum); no substrate: the methods are replaced
 * with the Objective-C runtime, the original implementations called through.
 *
 * SPDX-License-Identifier: MIT
 */
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CoreGraphics.h>
#import <IOSurface/IOSurfaceRef.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ---------------------------------------------------------------------------------------------------------------- */
/* configuration */

struct bf_config {
	int mode;
	double scale;          /* <= 0: the mode's default */
	double maxMs, slackMs;
	char recipes[256];
	int dump, quiet;
};

static struct bf_config bf_cfg = { 0, 0, 1500, 250, "modulesBackground,homeScreenOverlay", 0, 0 };
static CFTimeInterval bf_cfg_read_at = -1e9;
static int bf_sessions_live;   /* sessions running (main thread only): the cheap test every hook makes first */

static void
bf_say(BOOL error, const char *fmt, ...)
{
	char line[1024];
	va_list ap;
	int n, fd;

	if (!error && bf_cfg.quiet) return;
	n = snprintf(line, sizeof(line), "blurfreeze[%d]: ", getpid());
	va_start(ap, fmt);
	vsnprintf(line + n, sizeof(line) - (size_t)n, fmt, ap);
	va_end(ap);
	n = (int)strnlen(line, sizeof(line) - 2);
	line[n++] = '\n';
	fd = open("/dev/console", O_WRONLY | O_NOCTTY);
	(void)write(fd >= 0 ? fd : 2, line, (size_t)n);
	if (fd >= 0) close(fd);
}

static void
bf_read_config(void)
{
	char buf[512] = "";
	struct bf_config c = { 0, 0, 1500, 250, "modulesBackground,homeScreenOverlay", 0, 0 };
	int fd = open("/var/tmp/blurfreeze.args", O_RDONLY);

	if (fd >= 0) {
		ssize_t got = read(fd, buf, sizeof(buf) - 1);

		close(fd);
		buf[got > 0 ? got : 0] = '\0';
	}
	for (char *save = NULL, *w = strtok_r(buf, " \t\r\n", &save); w != NULL; w = strtok_r(NULL, " \t\r\n", &save)) {
		if (strncmp(w, "mode=", 5) == 0) c.mode = atoi(w + 5);
		else if (strncmp(w, "scale=", 6) == 0) c.scale = atof(w + 6);
		else if (strncmp(w, "maxms=", 6) == 0) c.maxMs = atof(w + 6);
		else if (strncmp(w, "slackms=", 8) == 0) c.slackMs = atof(w + 8);
		else if (strncmp(w, "recipes=", 8) == 0) strlcpy(c.recipes, w + 8, sizeof(c.recipes));
		else if (strncmp(w, "dump=", 5) == 0) c.dump = atoi(w + 5);
		else if (strncmp(w, "quiet=", 6) == 0) c.quiet = atoi(w + 6);
	}
	if (c.mode < 0 || c.mode > 2) c.mode = 0;
	if (c.scale > 1) c.scale = 1;
	if (c.maxMs < 50) c.maxMs = 50;
	if (c.maxMs > 5000) c.maxMs = 5000;
	if (c.slackMs < 30) c.slackMs = 30;
	if (memcmp(&c, &bf_cfg, sizeof(c)) != 0) {
		bf_cfg = c;
		bf_say(NO, "args: mode=%d scale=%g maxms=%g slackms=%g recipes=%s dump=%d", c.mode, c.scale, c.maxMs, c.slackMs,
		       c.recipes, c.dump);
	}
}

/* the switch, re-read at most once a second and never in the middle of a session */
static int
bf_mode(void)
{
	const CFTimeInterval now = CACurrentMediaTime();

	if (bf_sessions_live == 0 && now - bf_cfg_read_at > 1.0) {
		bf_cfg_read_at = now;
		bf_read_config();
	}
	return bf_cfg.mode;
}

static BOOL
bf_recipe_allowed(NSString *recipe)
{
	if (recipe == nil) return NO;
	for (NSString *r in [[NSString stringWithUTF8String:bf_cfg.recipes] componentsSeparatedByString:@","]) {
		if ([r isEqualToString:@"*"] || [r isEqualToString:recipe]) return YES;
	}
	return NO;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the private API, found at run time: nothing here links against a symbol iOS 16 might not have */

static BOOL (*bf_CARenderServerSnapshot)(mach_port_t, CFDictionaryRef);
static CFStringRef bf_kMode, bf_kModeExclude, bf_kContextList, bf_kContextId, bf_kDisplayName, bf_kDestination, bf_kTransform;
static Class bf_MTMaterialLayer, bf_CABackdropLayer;

static CFStringRef
bf_const(const char *name)
{
	CFStringRef *p = dlsym(RTLD_DEFAULT, name);

	return p != NULL ? *p : NULL;
}

static BOOL
bf_resolve(void)
{
	bf_CARenderServerSnapshot = dlsym(RTLD_DEFAULT, "CARenderServerSnapshot");
	bf_kMode = bf_const("kCASnapshotMode");
	bf_kModeExclude = bf_const("kCASnapshotModeExcludeContextList");
	bf_kContextList = bf_const("kCASnapshotContextList");
	bf_kContextId = bf_const("kCASnapshotContextId");
	bf_kDisplayName = bf_const("kCASnapshotDisplayName");
	bf_kDestination = bf_const("kCASnapshotDestination");
	bf_kTransform = bf_const("kCASnapshotTransform");
	bf_MTMaterialLayer = objc_getClass("MTMaterialLayer");
	bf_CABackdropLayer = objc_getClass("CABackdropLayer");
	if (bf_CARenderServerSnapshot == NULL || bf_kMode == NULL || bf_kModeExclude == NULL || bf_kContextList == NULL || bf_kContextId == NULL ||
	    bf_kDisplayName == NULL || bf_kDestination == NULL || bf_kTransform == NULL || bf_MTMaterialLayer == Nil ||
	    bf_CABackdropLayer == Nil) {
		bf_say(YES, "not armed: snapshot=%p mode=%p exclude=%p list=%p display=%p dest=%p transform=%p MTMaterialLayer=%p",
		       bf_CARenderServerSnapshot, bf_kMode, bf_kModeExclude, bf_kContextList, bf_kDisplayName, bf_kDestination,
		       bf_kTransform, (__bridge void *)bf_MTMaterialLayer);
		return NO;
	}
	return YES;
}

static id
bf_send(id obj, const char *sel)
{
	SEL s = sel_registerName(sel);

	return obj != nil && [obj respondsToSelector:s] ? ((id (*)(id, SEL))objc_msgSend)(obj, s) : nil;
}

static unsigned
bf_context_id(CALayer *layer)
{
	id context = bf_send(layer, "context");   /* -[CALayer context]: the CAContext of its tree (MaterialKit asks the same) */
	SEL s = sel_registerName("contextId");

	return context != nil && [context respondsToSelector:s] ? ((unsigned (*)(id, SEL))objc_msgSend)(context, s) : 0;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the crash guard: /var/tmp/blurfreeze.armed while a session (or its snapshot) is in progress */

#define BF_ARMED   "/var/tmp/blurfreeze.armed"
#define BF_TRIPPED "/var/tmp/blurfreeze.tripped"

static void
bf_guard_enter(int number)
{
	char line[64];
	int fd = open(BF_ARMED, O_WRONLY | O_CREAT | O_TRUNC, 0644);

	if (fd < 0) return;
	snprintf(line, sizeof(line), "session=%d pid=%d\n", number, getpid());
	(void)write(fd, line, strlen(line));
	fsync(fd);
	close(fd);
}

static void
bf_guard_leave(void)
{
	if (bf_sessions_live == 0) unlink(BF_ARMED);
}

/* at load: NO when the last SpringBoard died in a session or a trip has not been cleared (then nothing is installed) */
static BOOL
bf_guard_ok(void)
{
	char old[64] = "", prev[64] = "";
	int trips = 0, fd;

	if ((fd = open(BF_TRIPPED, O_RDONLY)) >= 0) {
		ssize_t got = read(fd, prev, sizeof(prev) - 1);

		close(fd);
		prev[got > 0 ? got : 0] = '\0';
		sscanf(prev, "trips=%d", &trips);
	}
	if ((fd = open(BF_ARMED, O_RDONLY)) >= 0) {
		ssize_t got = read(fd, old, sizeof(old) - 1);
		char line[160];

		close(fd);
		old[got > 0 ? got : 0] = '\0';
		old[strcspn(old, "\n")] = '\0';
		unlink(BF_ARMED);
		trips++;
		if ((fd = open(BF_TRIPPED, O_WRONLY | O_CREAT | O_TRUNC, 0644)) >= 0) {
			snprintf(line, sizeof(line), "trips=%d last=%s\n", trips, old);
			(void)write(fd, line, strlen(line));
			fsync(fd);
			close(fd);
		}
		bf_say(YES, "DISABLED: the last SpringBoard died during a session (%s); trips %d. rm " BF_TRIPPED " to re-arm", old, trips);
		return NO;
	}
	if (trips > 0 || prev[0] != '\0') {
		bf_say(YES, "DISABLED: " BF_TRIPPED " exists (%d trips); rm it to re-arm", trips);
		return NO;
	}
	return YES;
}

/* something of MATERIAL's own CAContext is drawn before it (a visible earlier sibling of it or of an ancestor): then
 * leaving its context out of the snapshot would leave out what is behind it too (the home screen's icons behind the
 * homeScreenOverlay material) */
static BOOL
bf_shares_context_behind(CALayer *m)
{
	for (CALayer *l = m, *p = m.superlayer; p != nil; l = p, p = p.superlayer) {
		for (CALayer *sib in p.sublayers) {
			if (sib == l) break;
			if (!sib.hidden && sib.opacity > 0) return YES;
		}
	}
	return NO;
}

/* the layers drawn over MATERIAL in its own context: it and every later sibling of it and of its ancestors */
static NSArray *
bf_layers_from(CALayer *m)
{
	NSMutableArray *out = [NSMutableArray arrayWithObject:m];

	for (CALayer *l = m, *p = m.superlayer; p != nil; l = p, p = p.superlayer) {
		BOOL after = NO;

		for (CALayer *sib in p.sublayers) {
			if (after && !sib.hidden) [out addObject:sib];
			if (sib == l) after = YES;
		}
	}
	return out;
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the original implementations */

static void (*bf_orig_updateWeighting)(id, SEL);
static void (*bf_orig_addAnimation)(id, SEL, CAAnimation *, NSString *);
static void (*bf_orig_setFilters)(id, SEL, NSArray *);
static void (*bf_orig_setValueForKeyPath)(id, SEL, id, NSString *);
static void (*bf_orig_setEnabled)(id, SEL, BOOL);

/* ---------------------------------------------------------------------------------------------------------------- */
/* a session */

@interface BFSession : NSObject
@property (nonatomic, weak) CALayer *material;
@property (nonatomic, strong) CALayer *proxy;
@property (nonatomic) int mode, number;
@property (nonatomic) CFTimeInterval started, deadline;
@property (nonatomic, strong) NSArray *savedFilters;   /* mode 2: the filters the material is to have back */
@property (nonatomic) BOOL savedEnabled;
@property (nonatomic) unsigned changes, animations;
@end
@implementation BFSession
@end

static const void *bf_session_key = &bf_session_key;
static int bf_session_count;

static BFSession *
bf_session(id layer)
{
	/* (sessions are the main thread's: a layer changed on another thread is never in one) */
	return bf_sessions_live > 0 && pthread_main_np() ? objc_getAssociatedObject(layer, bf_session_key) : nil;
}

/* the material's rectangle on the screen, in points: through its window's layer, whose frame is the screen's */
static CGRect
bf_screen_rect(CALayer *material, CALayer **rootOut)
{
	CALayer *root = material;

	while (root.superlayer != nil) root = root.superlayer;
	if (rootOut != NULL) *rootOut = root;
	return [material convertRect:material.bounds toLayer:root];
}

/* the contexts the snapshot leaves out: the material's own (unless OWN is NO), and every SpringBoard window's at or above
 * its window -- as context IDs (NSNumber); bf_snapshot makes them the dictionaries CoreAnimation wants */
static NSArray *
bf_excluded_contexts(CALayer *material, unsigned materialContext, BOOL own)
{
	NSMutableArray *ids = own ? [NSMutableArray arrayWithObject:@(materialContext)] : [NSMutableArray array];
	Class app = objc_getClass("UIApplication");
	id shared = app != Nil ? ((id (*)(id, SEL))objc_msgSend)(app, sel_registerName("sharedApplication")) : nil;
	NSArray *windows = bf_send(shared, "windows");
	double level = -1e9;

	for (id w in windows) {
		if (bf_context_id([w layer]) == materialContext) {
			level = MAX(level, ((double (*)(id, SEL))objc_msgSend)(w, sel_registerName("windowLevel")));
		}
	}
	for (id w in windows) {
		const unsigned cid = bf_context_id([w layer]);
		const double l = ((double (*)(id, SEL))objc_msgSend)(w, sel_registerName("windowLevel"));
		const BOOL hidden = ((BOOL (*)(id, SEL))objc_msgSend)(w, sel_registerName("isHidden"));

		if (cid != 0 && cid != materialContext && !hidden && l >= level && ![ids containsObject:@(cid)]) [ids addObject:@(cid)];
	}
	return ids;
}

static IOSurfaceRef
bf_surface(size_t width, size_t height)
{
	const size_t bpr = (width * 4 + 63) & ~(size_t)63;
	NSDictionary *p = @{
		(__bridge NSString *)kIOSurfaceWidth: @(width), (__bridge NSString *)kIOSurfaceHeight: @(height),
		(__bridge NSString *)kIOSurfaceBytesPerElement: @4, (__bridge NSString *)kIOSurfaceBytesPerRow: @(bpr),
		(__bridge NSString *)kIOSurfaceAllocSize: @(bpr * height), (__bridge NSString *)kIOSurfacePixelFormat: @((uint32_t)'BGRA'),
	};

	return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

/* what is behind MATERIAL on the screen, at SCALE of the screen's pixels, into a new surface (NULL: none) */
static IOSurfaceRef
bf_snapshot(CALayer *material, double scale, CGRect rect, NSArray *excluded, double *msOut, size_t *wOut, size_t *hOut)
{
	Class display = objc_getClass("CADisplay");
	id main = display != Nil ? ((id (*)(id, SEL))objc_msgSend)(display, sel_registerName("mainDisplay")) : nil;
	NSString *name = bf_send(main, "name");
	const double px = material.contentsScale > 0 ? material.contentsScale : 2.0;   /* points to pixels (2 on this phone) */
	const double s = px * scale;
	const size_t w = (size_t)MAX(1.0, ceil(rect.size.width * s)), h = (size_t)MAX(1.0, ceil(rect.size.height * s));
	IOSurfaceRef surface;
	CATransform3D t;
	CFTimeInterval began;
	BOOL ok;

	if (name == nil) {
		bf_say(YES, "no main display");
		return NULL;
	}
	surface = bf_surface(w, h);
	if (surface == NULL) {
		bf_say(YES, "IOSurfaceCreate %zux%zu failed", w, h);
		return NULL;
	}
	/* as MaterialKit's own snapshot: the display's points moved by the rectangle's origin, then scaled to pixels */
	t = CATransform3DConcat(CATransform3DMakeTranslation(-rect.origin.x, -rect.origin.y, 0), CATransform3DMakeScale(s, s, 1));
	/* Each element of kCASnapshotContextList is a DICTIONARY, not a number: QuartzCore 20H392's
	 * get_context_list_info(NSArray *, unsigned *, CATransform3D *) (0x188369fd4) enumerates the list and sends each
	 * element -objectForKey:kCASnapshotContextId ("contextId") -unsignedIntValue, and, when asked for transforms (the
	 * capture paths), -objectForKey:kCASnapshotTransform ("transform") -CATransform3DValue. With an include or exclude
	 * mode CARenderServerSnapshot_ (0x1882adc90) calls it without transforms and gives _CARenderServerSnapshotDisplay
	 * the dictionary's top-level kCASnapshotTransform. An NSNumber element raised "unrecognized selector
	 * -objectForKey:" inside the render call (SpringBoard aborted on the phone). */
	{
		NSMutableArray *list = [NSMutableArray arrayWithCapacity:excluded.count];

		for (NSNumber *n in excluded) [list addObject:@{ (__bridge NSString *)bf_kContextId: n }];
		began = CACurrentMediaTime();
		@try {
			ok = bf_CARenderServerSnapshot(0, (__bridge CFDictionaryRef)@{
				(__bridge NSString *)bf_kMode: (__bridge NSString *)bf_kModeExclude,
				(__bridge NSString *)bf_kContextList: list,
				(__bridge NSString *)bf_kDisplayName: name,
				(__bridge NSString *)bf_kDestination: (__bridge id)surface,
				(__bridge NSString *)bf_kTransform: [NSValue valueWithCATransform3D:t],
			});
		} @catch (NSException *e) {
			bf_say(YES, "CARenderServerSnapshot raised %s: %s", e.name.UTF8String, e.reason.UTF8String);
			ok = NO;
		}
	}
	*msOut = (CACurrentMediaTime() - began) * 1e3;
	if (!ok) {
		bf_say(YES, "CARenderServerSnapshot failed (%zux%zu)", w, h);
		CFRelease(surface);
		return NULL;
	}
	*wOut = w;
	*hOut = h;
	return surface;
}

static void
bf_dump(IOSurfaceRef surface, int number)
{
	char path[64];
	int fd;

	snprintf(path, sizeof(path), "/var/tmp/blurfreeze-%d.raw", number);
	fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) return;
	IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
	{
		const uint8_t *base = IOSurfaceGetBaseAddress(surface);
		const size_t bpr = IOSurfaceGetBytesPerRow(surface), w = IOSurfaceGetWidth(surface), h = IOSurfaceGetHeight(surface);

		for (size_t y = 0; y < h; y++) (void)write(fd, base + y * bpr, w * 4);
	}
	IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
	close(fd);
	bf_say(NO, "session %d: snapshot written to %s", number, path);
}

/* the material's filters as the snapshot layer takes them: copies, the blur normalising its edges (a backdrop replicates
 * the edge of its capture; a layer's blur would otherwise darken towards a transparent outside) */
static NSArray *
bf_proxy_filters(NSArray *filters)
{
	NSMutableArray *out = [NSMutableArray arrayWithCapacity:filters.count];

	for (id f in filters) {
		id c = [f respondsToSelector:@selector(copyWithZone:)] ? [f copy] : f;
		NSString *type = bf_send(c, "type");

		if ([type isEqualToString:@"gaussianBlur"]) {
			@try {
				[c setValue:@YES forKey:@"inputNormalizeEdges"];
			} @catch (NSException *e) {
				(void)e;
			}
		}
		[out addObject:c];
	}
	return out;
}

/* the animations of the material that the snapshot layer has to run too in mode 2: filters, opacity, zoom */
static BOOL
bf_mirrored_keypath(NSString *keyPath)
{
	return [keyPath hasPrefix:@"filters."] || [keyPath isEqualToString:@"opacity"] || [keyPath isEqualToString:@"zoom"];
}

static void
bf_mirror_animation(BFSession *s, CAAnimation *anim, NSString *key)
{
	CAPropertyAnimation *p;
	NSString *keyPath;

	if (![anim isKindOfClass:[CAPropertyAnimation class]]) return;
	keyPath = ((CAPropertyAnimation *)anim).keyPath;
	if (!bf_mirrored_keypath(keyPath)) return;
	p = [anim copy];
	if ([keyPath isEqualToString:@"zoom"]) {
		/* [H] a backdrop's zoom Z shows its capture magnified by 1 + Z about the centre: a scale of the snapshot layer.
		 * Only the basic and spring animations' from/to/by values are converted. */
		if (![p isKindOfClass:[CABasicAnimation class]]) return;
		CABasicAnimation *b = (CABasicAnimation *)p;
		NSNumber *from = b.fromValue, *to = b.toValue, *by = b.byValue;

		b.keyPath = @"transform.scale";
		if (from != nil) b.fromValue = @(1.0 + from.doubleValue);
		if (to != nil) b.toValue = @(1.0 + to.doubleValue);
		if (by != nil) b.byValue = by;
	}
	[s.proxy addAnimation:p forKey:[@"bf." stringByAppendingString:key ?: keyPath]];
}

static void bf_end(BFSession *s, const char *why);

static void
bf_check(BFSession *s)
{
	const CFTimeInterval now = CACurrentMediaTime();
	CALayer *m = s.material;

	if (m == nil) {
		bf_end(s, "material gone");
		return;
	}
	if (now >= s.started + bf_cfg.maxMs / 1e3) {
		bf_end(s, "maxms");
		return;
	}
	if (now >= s.deadline) {
		bf_end(s, "quiet");
		return;
	}
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)((MIN(s.deadline, s.started + bf_cfg.maxMs / 1e3) - now) * 1e9) + 1000000),
		       dispatch_get_main_queue(), ^{ bf_check(s); });
}

static void
bf_end(BFSession *s, const char *why)
{
	CALayer *m = s.material;
	const CFTimeInterval now = CACurrentMediaTime();

	if (s.proxy == nil) return;   /* (ended already) */
	[CATransaction begin];
	[CATransaction setDisableActions:YES];
	if (m != nil && s.mode == 2) {
		bf_orig_setFilters(m, @selector(setFilters:), s.savedFilters);
		bf_orig_setEnabled(m, sel_registerName("setEnabled:"), s.savedEnabled);
		/* an animation of the filters still running (maxms reached in the middle of one) goes back to the material */
		for (NSString *key in s.proxy.animationKeys) {
			CAAnimation *a = [s.proxy animationForKey:key];

			if ([a isKindOfClass:[CAPropertyAnimation class]] && [((CAPropertyAnimation *)a).keyPath hasPrefix:@"filters."] &&
			    a.beginTime + a.duration > now) {
				bf_orig_addAnimation(m, @selector(addAnimation:forKey:), a, [key substringFromIndex:3]);
			}
		}
	}
	[s.proxy removeFromSuperlayer];
	[CATransaction commit];
	s.proxy = nil;
	if (m != nil) objc_setAssociatedObject(m, bf_session_key, nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	bf_sessions_live--;
	bf_guard_leave();
	bf_say(NO, "session %d ended (%s) after %.0f ms: %u changes, %u animations", s.number, why, (now - s.started) * 1e3, s.changes,
	       s.animations);
}

static BFSession *
bf_begin(CALayer *m, int mode)
{
	NSString *recipe = bf_send(m, "recipeName");
	const double scale = bf_cfg.scale > 0 ? bf_cfg.scale : mode == 1 ? 1.0 : 0.5;
	const unsigned cid = bf_context_id(m);
	CALayer *root = nil, *proxy;
	CGRect rect;
	NSArray *excluded;
	IOSurfaceRef surface;
	double ms = 0;
	size_t w = 0, h = 0;
	BOOL shared;
	BFSession *s;

	if (!bf_recipe_allowed(recipe) || m.superlayer == nil || cid == 0) return nil;
	rect = bf_screen_rect(m, &root);
	if (CGRectIsEmpty(rect)) return nil;
	bf_guard_enter(bf_session_count + 1);
	shared = bf_shares_context_behind(m);
	excluded = bf_excluded_contexts(m, cid, !shared);
	if (!shared) {
		surface = bf_snapshot(m, scale, rect, excluded, &ms, &w, &h);
	} else {
		/* "hide": the material's context stays in the snapshot, with the material and what its context draws over it
		 * hidden for the moment of the snapshot (committed by a flush; at the start of a transition the material is
		 * nearly transparent, but one frame without it may reach the screen) */
		NSArray *over = bf_layers_from(m);

		[CATransaction begin];
		[CATransaction setDisableActions:YES];
		for (CALayer *l in over) l.hidden = YES;
		[CATransaction commit];
		[CATransaction flush];
		surface = bf_snapshot(m, scale, rect, excluded, &ms, &w, &h);
		[CATransaction begin];
		[CATransaction setDisableActions:YES];
		for (CALayer *l in over) l.hidden = NO;
		[CATransaction commit];
	}
	if (surface == NULL) {
		bf_guard_leave();
		return nil;
	}

	s = [BFSession new];
	s.material = m;
	s.mode = mode;
	s.number = ++bf_session_count;
	s.started = CACurrentMediaTime();
	s.deadline = s.started + bf_cfg.slackMs / 1e3;

	[CATransaction begin];
	[CATransaction setDisableActions:YES];
	proxy = [CALayer layer];
	proxy.name = @"blurfreeze";
	proxy.contents = (__bridge id)surface;
	proxy.contentsGravity = kCAGravityResize;
	proxy.magnificationFilter = kCAFilterLinear;
	proxy.minificationFilter = kCAFilterLinear;
	proxy.bounds = m.bounds;
	proxy.anchorPoint = m.anchorPoint;
	proxy.position = m.position;
	proxy.transform = m.transform;
	proxy.zPosition = m.zPosition;
	proxy.cornerRadius = m.cornerRadius;
	proxy.masksToBounds = m.masksToBounds || m.cornerRadius > 0;
	s.proxy = proxy;
	if (mode == 1) {
		proxy.opaque = YES;   /* what lets CoreAnimation leave out the layers under it, if it does */
	} else {
		NSArray *filters = m.filters;

		s.savedFilters = filters;
		s.savedEnabled = ((BOOL (*)(id, SEL))objc_msgSend)(m, sel_registerName("isEnabled"));
		proxy.opacity = m.opacity;
		proxy.filters = bf_proxy_filters(filters);
		{
			NSNumber *zoom = [m valueForKey:@"zoom"];

			if (zoom != nil && zoom.doubleValue != 0) proxy.transform = CATransform3DScale(m.transform, 1 + zoom.doubleValue, 1 + zoom.doubleValue, 1);
		}
		for (NSString *key in m.animationKeys) bf_mirror_animation(s, [m animationForKey:key], key);
		bf_orig_setFilters(m, @selector(setFilters:), nil);
		bf_orig_setEnabled(m, sel_registerName("setEnabled:"), NO);
	}
	[m.superlayer insertSublayer:proxy below:m];
	[CATransaction commit];
	CFRelease(surface);   /* (the layer keeps it) */

	objc_setAssociatedObject(m, bf_session_key, s, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	bf_sessions_live++;
	{
		NSMutableString *ex = [NSMutableString string];
		NSMutableString *fl = [NSMutableString string];

		for (NSNumber *n in excluded) [ex appendFormat:@"%@%u", ex.length ? @"," : @"", n.unsignedIntValue];
		for (id f in m.filters ?: s.savedFilters) [fl appendFormat:@"%@%@", fl.length ? @"," : @"", bf_send(f, "name") ?: @"?"];
		bf_say(NO, "session %d mode %d recipe %s strategy %s rect %.0f,%.0f %.0fx%.0f pt: snapshot %zux%zu (scale %g) in %.1f ms, excluded "
		       "contexts [%s] of context %u; material %s sublayers %lu filters [%s] animations %lu", s.number, mode, recipe.UTF8String, shared ? "hide" : "exclude",
		       rect.origin.x, rect.origin.y, rect.size.width, rect.size.height, w, h, scale, ms, ex.UTF8String, cid,
		       class_getName(object_getClass(m)), (unsigned long)m.sublayers.count, fl.UTF8String, (unsigned long)m.animationKeys.count);
	}
	if (bf_cfg.dump) bf_dump((__bridge IOSurfaceRef)proxy.contents, s.number);
	bf_check(s);
	return s;
}

/* a change of MATERIAL's weighting (UNTIL 0) or an animation of it that ends at UNTIL */
static void
bf_touch(CALayer *m, CFTimeInterval until, BOOL animation)
{
	BFSession *s;
	int mode;

	if (!pthread_main_np()) return;
	mode = bf_mode();
	s = bf_session(m);
	if (s == nil) {
		if (mode == 0) return;
		s = bf_begin(m, mode);
		if (s == nil) return;
	}
	if (animation) s.animations++;
	else s.changes++;
	s.deadline = MAX(s.deadline, MAX(until, CACurrentMediaTime()) + bf_cfg.slackMs / 1e3);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* the replacements */

/* dump=1: a material's weighting change, at most one line per material every 100 ms (the last material and time kept) */
static void
bf_dump_weighting(CALayer *m)
{
	static __unsafe_unretained CALayer *last;
	static CFTimeInterval lastAt;
	const CFTimeInterval now = CACurrentMediaTime();

	if (m == last && now - lastAt < 0.1) return;
	last = m;
	lastAt = now;
	bf_say(NO, "weighting %.3f on %s %p recipe=%s context=%u session=%d", [[m valueForKey:@"weighting"] doubleValue],
	       object_getClassName(m), (__bridge void *)m, [[bf_send(m, "recipeName") description] UTF8String], bf_context_id(m),
	       bf_session(m) != nil ? bf_session(m).number : 0);
}

static void
bf_updateWeighting(id self, SEL _cmd)
{
	bf_orig_updateWeighting(self, _cmd);
	if (!pthread_main_np()) return;
	(void)bf_mode();   /* (re-reads the switches, at most once a second, outside sessions) */
	if (bf_cfg.dump) bf_dump_weighting(self);
	bf_touch(self, 0, NO);
}

static CFTimeInterval
bf_animation_end(CAAnimation *a)
{
	double d = a.duration;
	SEL settling = sel_registerName("settlingDuration");

	if ([a respondsToSelector:settling]) d = MAX(d, ((double (*)(id, SEL))objc_msgSend)(a, settling));
	if (a.repeatCount > 1) d *= a.repeatCount;
	return (a.beginTime > 0 ? a.beginTime : CACurrentMediaTime()) + d;
}

static void
bf_addAnimation(id self, SEL _cmd, CAAnimation *anim, NSString *key)
{
	BFSession *s;

	bf_orig_addAnimation(self, _cmd, anim, key);
	/* the cheap way out first: switched off with no logging (as last read, under a second ago), or not the main thread */
	if (bf_cfg.mode == 0 && !bf_cfg.dump && bf_sessions_live == 0 && CACurrentMediaTime() - bf_cfg_read_at < 1.0) return;
	if (!pthread_main_np()) return;
	if (bf_cfg.dump && [self isKindOfClass:bf_MTMaterialLayer])
		bf_say(NO, "anim on %s recipe=%s key=%s keyPath=%s class=%s", object_getClassName(self),
		    [[bf_send(self, "recipeName") description] UTF8String], key.UTF8String,
		    [anim isKindOfClass:[CAPropertyAnimation class]] ? ((CAPropertyAnimation *)anim).keyPath.UTF8String : "-",
		    object_getClassName(anim));
	if (![self isKindOfClass:bf_MTMaterialLayer] || ![anim isKindOfClass:[CAPropertyAnimation class]]) return;
	{
		NSString *keyPath = ((CAPropertyAnimation *)anim).keyPath;

		if (![keyPath isEqualToString:@"weighting"] && !bf_mirrored_keypath(keyPath)) return;
	}
	bf_touch(self, bf_animation_end(anim), YES);
	s = bf_session(self);
	if (s != nil && s.mode == 2) bf_mirror_animation(s, anim, key);
}

static void
bf_setFilters(id self, SEL _cmd, NSArray *filters)
{
	BFSession *s = bf_session(self);

	if (s != nil && s.mode == 2) {
		s.savedFilters = filters;
		s.proxy.filters = bf_proxy_filters(filters);
		return;
	}
	bf_orig_setFilters(self, _cmd, filters);
}

static void
bf_setValueForKeyPath(id self, SEL _cmd, id value, NSString *keyPath)
{
	BFSession *s = bf_session(self);

	if (s != nil && s.mode == 2 && [keyPath hasPrefix:@"filters."]) {
		/* the material has no filters during the session: the value goes to the snapshot layer and to the copies the
		 * material gets back at the end */
		NSArray *parts = [keyPath componentsSeparatedByString:@"."];

		[s.proxy setValue:value forKeyPath:keyPath];
		if (parts.count == 3) {
			for (id f in s.savedFilters) {
				if ([bf_send(f, "name") isEqual:parts[1]]) [f setValue:value forKey:parts[2]];
			}
		}
		return;
	}
	bf_orig_setValueForKeyPath(self, _cmd, value, keyPath);
	if (s != nil && s.mode == 2 && [keyPath isEqualToString:@"opacity"]) s.proxy.opacity = [value floatValue];
}

static void
bf_setEnabled(id self, SEL _cmd, BOOL enabled)
{
	BFSession *s = bf_session(self);

	if (s != nil && s.mode == 2) {
		s.savedEnabled = enabled;
		return;
	}
	bf_orig_setEnabled(self, _cmd, enabled);
}

static BOOL
bf_replace(Class cls, const char *selector, IMP replacement, void *original)
{
	Method m = cls != Nil ? class_getInstanceMethod(cls, sel_registerName(selector)) : NULL;

	if (m == NULL) {
		bf_say(YES, "not armed: %s has no -%s", cls != Nil ? class_getName(cls) : "(nil)", selector);
		return NO;
	}
	/* a class that inherits the method gets one of its own first, so that the replacement is the class's alone */
	class_addMethod(cls, method_getName(m), method_getImplementation(m), method_getTypeEncoding(m));
	m = class_getInstanceMethod(cls, sel_registerName(selector));
	*(IMP *)original = method_setImplementation(m, replacement);
	return YES;
}

__attribute__((constructor))
static void
bf_init(void)
{
	const char *prog = getprogname();

	if (prog == NULL || strcmp(prog, "SpringBoard") != 0) return;
	/* CoreMaterial is SpringBoard's already, but a library loaded early may come before it */
	dlopen("/System/Library/PrivateFrameworks/CoreMaterial.framework/CoreMaterial", RTLD_LAZY);
	if (!bf_guard_ok() || !bf_resolve()) return;
	if (!bf_replace(bf_MTMaterialLayer, "_updateForChangeInWeighting", (IMP)bf_updateWeighting, &bf_orig_updateWeighting) ||
	    !bf_replace([CALayer class], "addAnimation:forKey:", (IMP)bf_addAnimation, &bf_orig_addAnimation) ||
	    !bf_replace([CALayer class], "setFilters:", (IMP)bf_setFilters, &bf_orig_setFilters) ||
	    !bf_replace([CALayer class], "setValue:forKeyPath:", (IMP)bf_setValueForKeyPath, &bf_orig_setValueForKeyPath) ||
	    !bf_replace(bf_CABackdropLayer, "setEnabled:", (IMP)bf_setEnabled, &bf_orig_setEnabled)) {
		return;
	}
	bf_read_config();
	bf_cfg_read_at = CACurrentMediaTime();
	bf_say(YES, "armed (mode %d now; /var/tmp/blurfreeze.args switches it)", bf_cfg.mode);
}

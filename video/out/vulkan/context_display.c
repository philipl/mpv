/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <fcntl.h>
#include <unistd.h>

#include "libmpv/render_gl.h"
#include "options/m_config.h"
#include "osdep/timer.h"
#include "video/out/drm_common.h"

#include "context.h"
#include "utils.h"

struct vulkan_display_opts {
    char *display_spec;
};

static bool parse_display_spec(struct mp_log *log,
                               const char *display_spec,
                               int *display, int *mode, int *plane)
{
    int parsed_display = 0;
    int parsed_mode = 0;
    int parsed_plane = 0;
    bool ret = false;

    if (!display_spec) {
        goto finish;
    }
    const char *mode_ptr = strchr(display_spec, ':');
    if (mode_ptr) {
        parsed_display = atoi(display_spec);
        const char *plane_ptr = strchr(mode_ptr + 1, ':');
        if (plane_ptr) {
            parsed_mode = atoi(mode_ptr + 1);
        }
        if (*(plane_ptr + 1) != '\0') {
            parsed_plane = atoi(plane_ptr + 1);
        }

        ret = true;
    }
    mp_dbg(log, "Parsed Display Spec: %d, %d, %d\n",
           parsed_display, parsed_mode, parsed_plane);


  finish:
    if (display) {
        *display = parsed_display;
    }
    if (mode) {
        *mode = parsed_mode;
    }
    if (plane) {
        *plane = parsed_plane;
    }
    return ret;
}

static bool print_display_info(struct mp_log *log) {
    bool ret = false;
    VkResult res;

    // Use a dummy as parent for all other allocations.
    void *tmp = talloc_new(NULL);

    // Create a dummy instance to list the resources
    VkInstanceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = (const char*[]) {
            VK_KHR_DISPLAY_EXTENSION_NAME
        },
    };

    VkInstance inst = NULL;
    res = vkCreateInstance(&info, NULL, &inst);
    if (res != VK_SUCCESS) {
        mp_warn(log, "Unable to create Vulkan instance.\n");
        goto done;
    }

    uint32_t num_devices = 0;
    vkEnumeratePhysicalDevices(inst, &num_devices, NULL);
    if (!num_devices) {
        mp_info(log, "No Vulkan devices detected.\n");
        ret = true;
        goto done;
    }

    VkPhysicalDevice *devices = talloc_array(tmp, VkPhysicalDevice, num_devices);
    vkEnumeratePhysicalDevices(inst, &num_devices, devices);
    if (res != VK_SUCCESS) {
        mp_warn(log, "Failed enumerating physical devices.\n");
        goto done;
    }

    mp_info(log, "Vulkan Devices:\n");
    for (int i = 0; i < num_devices; i++) {
        VkPhysicalDevice device = devices[i];

        VkPhysicalDeviceProperties prop;
        vkGetPhysicalDeviceProperties(device, &prop);
        mp_info(log, "  '%s' (GPU %d, ID %x:%x)\n", prop.deviceName, i,
                (unsigned)prop.vendorID, (unsigned)prop.deviceID);

        // Count displays. This must be done before enumerating planes with the
        // Intel driver, or it will not enumerate any planes. WTF.
        int num_displays = 0;
        vkGetPhysicalDeviceDisplayPropertiesKHR(device, &num_displays, NULL);
        if (!num_displays) {
            mp_info(log, "    No available displays for device.\n");
            continue;
        }

        // Enumerate Planes
        int num_planes = 0;
        vkGetPhysicalDeviceDisplayPlanePropertiesKHR(device, &num_planes, NULL);
        if (!num_planes) {
            mp_info(log, "    No available planes for device.\n");
            continue;
        }

        VkDisplayPlanePropertiesKHR *planes =
            talloc_array(tmp, VkDisplayPlanePropertiesKHR, num_planes);
        res = vkGetPhysicalDeviceDisplayPlanePropertiesKHR(device, &num_planes, planes);
        if (res != VK_SUCCESS) {
            mp_warn(log, "    Failed enumerating planes\n");
            continue;
        }

        VkDisplayKHR **planes_to_displays =
            talloc_array(tmp, VkDisplayKHR *, num_planes);
        for (int j = 0; j < num_planes; j++) {
            int num_displays_for_plane = 0;
            vkGetDisplayPlaneSupportedDisplaysKHR(device, j,
                                                  &num_displays_for_plane, NULL);
            if (!num_displays) {
                continue;
            }

            VkDisplayKHR *displays =
                talloc_array(planes_to_displays, VkDisplayKHR,
                             num_displays_for_plane + 1);
            res = vkGetDisplayPlaneSupportedDisplaysKHR(device, j,
                                                        &num_displays_for_plane,
                                                        displays);
            if (res != VK_SUCCESS) {
                mp_warn(log, "      Failed enumerating plane displays\n");
                continue;
            }
            planes_to_displays[j] = displays;
        }

        // Enumerate Displays and Modes
        VkDisplayPropertiesKHR *props =
            talloc_array(tmp, VkDisplayPropertiesKHR, num_displays);
        res = vkGetPhysicalDeviceDisplayPropertiesKHR(device, &num_displays, props);
        if (res != VK_SUCCESS) {
            mp_warn(log, "    Failed enumerating display properties\n");
            continue;
        }

        for (int j = 0; j < num_displays; j++) {
            mp_info(log, "    Display %d: '%s' (%dx%d)\n",
                    j,
                    props[j].displayName,
                    props[j].physicalResolution.width,
                    props[j].physicalResolution.height);

            VkDisplayKHR display = props[j].display;

            mp_info(log, "    Modes:\n");

            int num_modes = 0;
            vkGetDisplayModePropertiesKHR(device, display, &num_modes, NULL);
            if (!num_modes) {
                mp_info(log, "      No available modes for display.\n");
                continue;
            }

            VkDisplayModePropertiesKHR *modes =
                talloc_array(tmp, VkDisplayModePropertiesKHR, num_modes);
            res = vkGetDisplayModePropertiesKHR(device, display, &num_modes, modes);
            if (res != VK_SUCCESS) {
                mp_warn(log, "      Failed enumerating display modes\n");
                continue;
            }

            for (int k = 0; k < num_modes; k++) {
                mp_info(log, "      Mode %02d: %dx%d (%02d.%03d Hz)\n",
                        k,
                        modes[k].parameters.visibleRegion.width,
                        modes[k].parameters.visibleRegion.height,
                        modes[k].parameters.refreshRate / 1000,
                        modes[k].parameters.refreshRate % 1000);
            }

            mp_info(log, "    Planes:\n");
            for (int k = 0; k < num_planes; k++) {
                VkDisplayKHR *displays = planes_to_displays[k];
                for (int d = 0; displays[d]; d++) {
                    if (displays[d] == display) {
                        mp_info(log, "      Plane: %d\n", k);
                    }
                }
                displays = NULL;
            }
        }
    }

done:
    talloc_free(tmp);
    vkDestroyInstance(inst, NULL);
    return ret;
}

static int display_validate_spec(struct mp_log *log, const struct m_option *opt,
                                struct bstr name, struct bstr param)
{
    if (bstr_equals0(param, "help")) {
        print_display_info(log);
        return M_OPT_EXIT;
    }

    char *spec = bstrto0(NULL, param);
    if (!parse_display_spec(log, spec, NULL, NULL, NULL)) {
        mp_fatal(log, "Invalid value for option vulkan-display-spec. "
                 "Must be a string of the format 'D:M:P' where each of D, M, P "
                 "is the index of a Display, Mode, and Plane, or 'help'\n");
        talloc_free(spec);
        return M_OPT_INVALID;
    }
    talloc_free(spec);

    return 1;
}

#define OPT_BASE_STRUCT struct vulkan_display_opts
const struct m_sub_options vulkan_display_conf = {
    .opts = (const struct m_option[]) {
        OPT_STRING_VALIDATE("vulkan-display-spec", display_spec, 0, display_validate_spec),
        {0}
    },
    .size = sizeof(struct vulkan_display_opts),
    .defaults = &(struct vulkan_display_opts) {
        .display_spec = "0:0:0",
    },
};

struct priv {
    struct mpvk_ctx vk;
    struct vulkan_display_opts *opts;
    uint32_t width;
    uint32_t height;

    drmModeCrtc *old_crtc;

    bool vt_switcher_active;
    struct vt_switcher vt_switcher;

    struct mpv_opengl_drm_params_v2 drm_params;
};

static bool open_render_fd(struct ra_ctx *ctx, int kms_fd)
{
    struct priv *p = ctx->priv;
    p->drm_params.fd = -1;
    p->drm_params.render_fd = -1;

    char *render_path = drmGetRenderDeviceNameFromFd(kms_fd);
    int fd = open(render_path, O_RDWR | O_CLOEXEC);
    free(render_path);

    p->drm_params.render_fd = fd;
    return fd != -1;
}

static void crtc_save(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;

    struct kms *kms = kms_create(ctx->log,
                                 ctx->vo->opts->drm_opts->drm_connector_spec,
                                 ctx->vo->opts->drm_opts->drm_mode_spec,
                                 ctx->vo->opts->drm_opts->drm_draw_plane,
                                 ctx->vo->opts->drm_opts->drm_drmprime_video_plane,
                                 ctx->vo->opts->drm_opts->drm_atomic);
    if (!kms) {
        MP_WARN(ctx, "Failed to create KMS to save old crtc mode.\n");
        return;
    }

    p->old_crtc = drmModeGetCrtc(kms->fd, kms->crtc_id);
    if (!p->old_crtc) {
        MP_WARN(ctx, "Failed to save old crtc mode.\n");
    }

    open_render_fd(ctx, kms->fd);

    kms_destroy(kms);
}

static void crtc_release(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;

    if (p->old_crtc) {
        struct kms *kms = kms_create(ctx->log,
                                    ctx->vo->opts->drm_opts->drm_connector_spec,
                                    ctx->vo->opts->drm_opts->drm_mode_spec,
                                    ctx->vo->opts->drm_opts->drm_draw_plane,
                                    ctx->vo->opts->drm_opts->drm_drmprime_video_plane,
                                    ctx->vo->opts->drm_opts->drm_atomic);
        if (!kms) {
            MP_WARN(ctx, "Failed to create KMS to restore old crtc mode.\n");
            return;
        }

        int ret = drmModeSetCrtc(kms->fd,
                        p->old_crtc->crtc_id, p->old_crtc->buffer_id,
                        p->old_crtc->x, p->old_crtc->y,
                        &kms->connector->connector_id, 1,
                        &p->old_crtc->mode);
        drmModeFreeCrtc(p->old_crtc);
        if (ret != 0) {
            MP_WARN(ctx, "Failed to restore old crtc mode.\n");
        }
        p->old_crtc = NULL;

        kms_destroy(kms);
    }
}

static void release_vt(void *data)
{
    // TODO: Anything? crtc_save/restore doesn't work.
}

static void acquire_vt(void *data)
{
    // TODO: Anything? crtc_save/restore doesn't work.
}

static void display_uninit(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;

    ra_vk_ctx_uninit(ctx);
    mpvk_uninit(&p->vk);

    if (p->drm_params.fd != -1)
        close(p->drm_params.fd);
    if (p->drm_params.render_fd != -1)
        close(p->drm_params.render_fd);

    crtc_release(ctx);

    if (p->vt_switcher_active)
        vt_switcher_destroy(&p->vt_switcher);
}

static bool display_init(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv = talloc_zero(ctx, struct priv);
    struct mpvk_ctx *vk = &p->vk;
    int msgl = ctx->opts.probing ? MSGL_V : MSGL_ERR;
    VkResult res;
    bool ret = false;
    uint32_t num = 0;

    void *tmp = talloc_new(NULL);

    p->opts = mp_get_config_group(p, ctx->global, &vulkan_display_conf);
    int display_idx, mode_idx, plane_idx;
    parse_display_spec(ctx->log, p->opts->display_spec,
                       &display_idx, &mode_idx, &plane_idx);

    p->vt_switcher_active = vt_switcher_init(&p->vt_switcher, ctx->vo->log);
    if (p->vt_switcher_active) {
        vt_switcher_acquire(&p->vt_switcher, acquire_vt, ctx);
        vt_switcher_release(&p->vt_switcher, release_vt, ctx);
    } else {
        MP_WARN(ctx, "Failed to set up VT switcher. Terminal switching will be unavailable.\n");
    }

    crtc_save(ctx);

    if (!mpvk_init(vk, ctx, VK_KHR_DISPLAY_EXTENSION_NAME))
        goto error;

    char *device_name = ra_vk_ctx_get_device_name(ctx);
    struct pl_vulkan_device_params vulkan_params = {
        .instance = vk->vkinst->instance,
        .device_name = device_name,
    };
    VkPhysicalDevice device = pl_vulkan_choose_device(vk->ctx, &vulkan_params);
    talloc_free(device_name);
    if (!device) {
        MP_MSG(ctx, msgl, "Failed to open physical device.\n");
        goto error;
    }

    num = 0;
    vkGetPhysicalDeviceDisplayPropertiesKHR(device, &num, NULL);
    if (!num) {
        MP_MSG(ctx, msgl, "No available displays.\n");
        goto error;
    }
    if (display_idx + 1 > num) {
        MP_MSG(ctx, msgl, "Selected display (%d) not present.\n", display_idx);
        goto error;
    }

    VkDisplayPropertiesKHR *props =
        talloc_array(tmp, VkDisplayPropertiesKHR, num);
    res = vkGetPhysicalDeviceDisplayPropertiesKHR(device, &num, props);
    if (res != VK_SUCCESS) {
        MP_MSG(ctx, msgl, "Failed enumerating display properties\n");
        goto error;
    }

    VkDisplayKHR display = props[display_idx].display;

    num = 0;
    vkGetDisplayModePropertiesKHR(device, display, &num, NULL);
    if (!num) {
        MP_MSG(ctx, msgl, "No available modes.\n");
        goto error;
    }
    if (mode_idx + 1 > num) {
        MP_MSG(ctx, msgl, "Selected mode (%d) not present.\n", mode_idx);
        goto error;
    }

    VkDisplayModePropertiesKHR *modes =
        talloc_array(tmp, VkDisplayModePropertiesKHR, num);
    res = vkGetDisplayModePropertiesKHR(device, display, &num, modes);
    if (res != VK_SUCCESS) {
        MP_MSG(ctx, msgl, "Failed enumerating display modes\n");
        goto error;
    }

    VkDisplayModePropertiesKHR *mode = &modes[mode_idx];

    num = 0;
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR(device, &num, NULL);
    if (!num) {
        MP_MSG(ctx, msgl, "No available planes.\n");
        goto error;
    }
    if (plane_idx + 1 > num) {
        MP_MSG(ctx, msgl, "Selected plane (%d) not present.\n", plane_idx);
        goto error;
    }

    VkDisplaySurfaceCreateInfoKHR xinfo = {
        .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
        .displayMode = mode->displayMode,
        .imageExtent = mode->parameters.visibleRegion,
        .planeIndex = plane_idx,
        .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
    };

    res = vkCreateDisplayPlaneSurfaceKHR(vk->vkinst->instance, &xinfo, NULL,
                                         &vk->surface);
    if (res != VK_SUCCESS) {
        MP_MSG(ctx, msgl, "Failed creating Display surface\n");
        goto error;
    }

    p->width = mode->parameters.visibleRegion.width;
    p->height = mode->parameters.visibleRegion.height;

    struct ra_vk_ctx_params params = {0};
    if (!ra_vk_ctx_init(ctx, vk, params, VK_PRESENT_MODE_FIFO_KHR))
        goto error;

    ra_add_native_resource(ctx->ra, "drm_params_v2", &p->drm_params);

    ret = true;

done:
    talloc_free(tmp);
    return ret;

error:
    display_uninit(ctx);
    goto done;
}

static bool display_reconfig(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    return ra_vk_ctx_resize(ctx, p->width, p->height);
}

static int display_control(struct ra_ctx *ctx, int *events, int request, void *arg)
{
    return VO_NOTIMPL;
}

static void display_wakeup(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    if (p->vt_switcher_active)
        vt_switcher_interrupt_poll(&p->vt_switcher);
}

static void display_wait_events(struct ra_ctx *ctx, int64_t until_time_us)
{
    struct priv *p = ctx->priv;
    if (p->vt_switcher_active) {
        int64_t wait_us = until_time_us - mp_time_us();
        int timeout_ms = MPCLAMP((wait_us + 500) / 1000, 0, 10000);
        vt_switcher_poll(&p->vt_switcher, timeout_ms);
    } else {
        vo_wait_default(ctx->vo, until_time_us);
    }
}

const struct ra_ctx_fns ra_ctx_vulkan_display = {
    .type           = "vulkan",
    .name           = "displayvk",
    .reconfig       = display_reconfig,
    .control        = display_control,
    .wakeup         = display_wakeup,
    .wait_events    = display_wait_events,
    .init           = display_init,
    .uninit         = display_uninit,
};

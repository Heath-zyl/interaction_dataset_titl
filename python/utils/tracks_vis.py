#!/usr/bin/env python

import matplotlib
import matplotlib.patches
import matplotlib.transforms
import numpy as np

from .dataset_types import Track, MotionState


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=ms.psi_rad)


def polygon_xy_from_motionstate_pedest(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return np.array([lowleft, lowright, upright, upleft])


def update_objects_plot(timestamp, patches_dict, text_dict, axes, track_dict=None, pedest_dict=None):
    
    # print(timestamp, track_dict.keys())
    
    if track_dict is not None:

        for key, value in track_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)
                
                # print(key, ms, patches_dict.keys())
                
                if key not in patches_dict:
                    width = value.width
                    length = value.length

                    if isinstance(key, str) and '_auto' in key:
                        rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width*.8, length*.8), closed=True, zorder=20, color='black', alpha=1)
                        attn = None
                    elif isinstance(key, str) and 'ego' in key:
                        rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True, zorder=20, color=(42/255, 157/255, 142/255))
                        attn = None
                    else:
                        attn = ms.get_attn_weight()
                        if attn is not None:
                            r, g, b = 1., 1 - attn, 0.
                            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True, zorder=20, color=(r,g,b))
                        else:
                            rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True, zorder=20)
                    
                    patches_dict[key] = rect
                    axes.add_patch(rect)
                    
                    # if attn is None:
                    #     text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30)
                    # else:
                    #     text_dict[key] = axes.text(ms.x, ms.y + 2, str(key)+':%.4f'%attn, horizontalalignment='center', zorder=30)
                    
                    # text_dict[key] = axes.text(ms.x, ms.y - 1, str(key), horizontalalignment='center', zorder=30)
                    
                    if attn is not None:
                        text_dict[key] = axes.text(ms.x, ms.y - 1, str(key), horizontalalignment='center', zorder=30)
                        text_dict[key] = axes.text(ms.x, ms.y + 1, '%.3f'%attn, horizontalalignment='center', zorder=30, fontweight='bold')
                        # print(help(axes.text))
                        # text_dict[key] = axes.text(ms.x, ms.y + 1, str(attn)[:4], horizontalalignment='center', zorder=30)
                    else:
                        text_dict[key] = axes.text(ms.x, ms.y + 1, 'ego', horizontalalignment='center', zorder=30)
                
                else:
                    width = value.width
                    length = value.length
                    
                    if isinstance(key, str) and 'auto' in key:
                        patches_dict[key].set_xy(polygon_xy_from_motionstate(ms, width*.8, length*.8))
                        attn = None
                    elif isinstance(key, str) and 'ego' in key:
                        patches_dict[key].set_xy(polygon_xy_from_motionstate(ms, width, length))
                        attn = None
                    else:
                        patches_dict[key].set_xy(polygon_xy_from_motionstate(ms, width, length))
                        attn = ms.get_attn_weight()
                        if attn is not None:
                            r, g, b = 1., 1 - attn, 0.
                    
                    text_dict[key].set_position((ms.x, ms.y + 2))
                    if attn is not None:
                        text_dict[key].set_text(str(key)+':%.4f'%attn)
                        patches_dict[key].set_color((r,g,b))
                    
            else:
                if key in patches_dict:
                    patches_dict[key].remove()
                    patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)


    if pedest_dict is not None:

        for key, value in pedest_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                # object is visible
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in patches_dict:
                    width = 1.5
                    length = 1.5

                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate_pedest(ms, width, length),
                                                      closed=True, zorder=20, color='red')
                    patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 2, str(key), horizontalalignment='center', zorder=30)
                else:
                    width = 1.5
                    length = 1.5
                    patches_dict[key].set_xy(polygon_xy_from_motionstate_pedest(ms, width, length))
                    text_dict[key].set_position((ms.x, ms.y + 2))
            else:
                if key in patches_dict:
                    patches_dict[key].remove()
                    patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)

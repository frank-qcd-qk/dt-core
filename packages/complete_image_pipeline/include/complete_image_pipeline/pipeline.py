import yaml
from collections import OrderedDict

# Robotics fun
import cv2
import matplotlib
import numpy as np
from datetime import  datetime
# Default
import duckietown_utils as dtu

# Segmentation Toolkit
from duckietown_msgs.msg import Segment, SegmentList
from duckietown_segmaps.draw_map_on_images import plot_map, predict_segments
from duckietown_segmaps.maps import FRAME_AXLE, plot_map_and_segments, FRAME_GLOBAL
from duckietown_segmaps.transformations import TransformationsInfo

# Easy Algo
from easy_algo import get_easy_algo_db
from easy_node.utils.timing import ProcessingTimingStats, FakeContext

# GPG
from image_processing.calibration_utils import get_ground_projection
# lane_filter
from lane_filter.lane_filter import LaneFilterHistogram
# line_detector
from line_detector.line_detector import LineDetector
# anti_instagram
from image_processing.anti_instagram import AntiInstagram
# localization
from localization_templates import FAMILY_LOC_TEMPLATES


@dtu.contract(ground_truth='SE2|None', image='array[HxWx3](uint8)')
def run_pipeline(image,
                 all_details=False,
                 ground_truth=None,
                 actual_map=None,
                 quick=False):
    """
        Image: numpy (H,W,3) == BGR
        Returns a dictionary, res with the following fields:

            res['input_image']

        ground_truth = pose
    """

    print('backend: %s' % matplotlib.get_backend())
    print('fname: %s' % matplotlib.matplotlib_fname())

    # Verify Image
    dtu.check_isinstance(image, np.ndarray)
    printWtime("Image check passed")

    # Initial Setup
    vehicle_name = dtu.get_current_robot_name()
    res = OrderedDict()  # The image book that holds all the images
    stats = OrderedDict()  # The stats book that holds all the statistics
    # TODO: This is not a good way to specify all the configuration files... Need smarter way...
    master_path = "/code/catkin_ws/src/dt-core/packages/"
    gpg_config_path = master_path + "ground_projection/config/ground_projection_node/default.yaml"
    line_detector_config_path = master_path + "line_detector/config/line_detector_node/default.yaml"
    lane_filter_config_path = master_path + "lane_filter/config/lane_filter_node/default.yaml"
    anti_instagram_config_path = master_path + "anti_instagram/config/anti_instagram_node/default.yaml"
    printWtime("Record book initialization complete!")

    # Object Initialization
    # Ground Projection Init
    # TODO: This is using a decrepted call. This need update for ente refactor
    gp = get_ground_projection(vehicle_name)
    gpg = gp.get_ground_projection_geometry()
    with open(gpg_config_path) as file:
        gpg_config = yaml.load(file, Loader=yaml.FullLoader)
    printWtime("GPG load success!")

    # Line detector Init
    with open(line_detector_config_path) as file:
        line_detector_config = yaml.load(file, Loader=yaml.FullLoader)
    line_detector = LineDetector()
    printWtime("line_detector load success!")

    # Lane filter Init
    with open(lane_filter_config_path) as file:
        lane_filter_config = yaml.load(file, Loader=yaml.FullLoader)
    lane_filter = LaneFilterHistogram(**lane_filter_config["lane_filter_histogram_configuration"])
    printWtime("lane_filter load success!")

    # AntiInstagram Init
    with open(anti_instagram_config_path) as file:
        anti_instagram_config = yaml.load(file, Loader=yaml.FullLoader)
    ai = AntiInstagram()
    printWtime("anti_instagram load success!")

    # Analysis tools
    # pts = ProcessingTimingStats()
    # pts.reset()
    # pts.received_message()
    # pts.decided_to_process()
    # print("Analysis tool load success! ")

    # Got Image
    res['Raw_input_image'] = image
    printWtime("Image load complete!")
    # =====Anti Instagram===== #
    [lower, upper] = ai.calculate_color_balance_thresholds(image, anti_instagram_config["output_scale"],
                                                           anti_instagram_config["color_balance_scale"])
    printWtime("Calculate AI Transform complete!")
    transformed = ai.apply_color_balance(lower, upper, image)
    printWtime("Apply AI Color balance complete!")
    res['Balanced'] = transformed
    printWtime(+"AI Complete! ")
    # =====Anti Instagram===== #

    # =====Edge Detection===== #

    segment_list2 = image_prep.process(pts, image,
                                       line_detector, transform=ai.apply_color_balance)
    res['resized and corrected'] = image_prep.image_corrected


    dtu.logger.debug('segment_list2: %s' % len(segment_list2.segments))

    res['segments_on_image_input_transformed'] = \
            vs_fancy_display(image_prep.image_cv, segment_list2)

    res['segments_on_image_input_transformed_resized'] = \
            vs_fancy_display(image_prep.image_resized, segment_list2)

    grid = get_grid(image.shape[:2])
    res['grid'] = grid
    res['grid_remapped'] = gpg.rectify(grid)

    res['difference between the two'] = res['image_input']*0.5 + res['image_input_rect']*0.5

    with pts.phase('rectify_segments'):
        segment_list2_rect = rectify_segments(gpg, segment_list2)

    # Project to ground
    with pts.phase('find_ground_coordinates'):
        sg = find_ground_coordinates(gpg, segment_list2_rect)

    lane_filter.initialize()
    if all_details:
        res['prior'] = lane_filter.get_plot_phi_d()

    with pts.phase('lane filter update'):
        print(type(lane_filter).__name__)
        if type(lane_filter).__name__ == 'LaneFilterHistogram':
            # XXX merging pain
            _likelihood = lane_filter.update(sg.segments)
        else:
            _likelihood = lane_filter.update(sg)

    if not quick:
        with pts.phase('lane filter plot'):
            res['likelihood'] = lane_filter.get_plot_phi_d(ground_truth=ground_truth)
    easy_algo_db = get_easy_algo_db()

    if isinstance(lane_filter, LaneFilterMoreGeneric):
        template_name = lane_filter.localization_template
    else:
        template_name = 'DT17_template_straight_straight'
        dtu.logger.debug('Using default template %r for visualization' % template_name)

    localization_template = \
        easy_algo_db.create_instance(FAMILY_LOC_TEMPLATES, template_name)

    with pts.phase('lane filter get_estimate()'):
        est = lane_filter.get_estimate()

    # Coordinates in TILE frame
    g = localization_template.pose_from_coords(est)
    tinfo = TransformationsInfo()
    tinfo.add_transformation(frame1=FRAME_GLOBAL, frame2=FRAME_AXLE, g=g)

    if all_details:
        if actual_map is not None:
            #         sm_axle = tinfo.transform_map_to_frame(actual_map, FRAME_AXLE)
            res['real'] = plot_map_and_segments(actual_map, tinfo, sg.segments, dpi=120,
                                                ground_truth=ground_truth)

    with pts.phase('rectify'):
        rectified0 = gpg.rectify(image)

    rectified = ai.apply_color_balance(rectified0)

    if all_details:
        res['image_input_rect'] = rectified

    res['segments rectified on image rectified'] = \
        vs_fancy_display(rectified, segment_list2_rect)

    assumed = localization_template.get_map()

    if not quick:
        with pts.phase('plot_map_and_segments'):
            res['model assumed for localization'] = plot_map_and_segments(assumed, tinfo, sg.segments, dpi=120,
                                                                          ground_truth=ground_truth)

    assumed_axle = tinfo.transform_map_to_frame(assumed, FRAME_AXLE)

    with pts.phase('plot_map reprojected'):
        res['map reprojected on image'] = plot_map(rectified, assumed_axle, gpg,
                                                   do_ground=False, do_horizon=True,
                                                   do_faces=False, do_faces_outline=True,
                                                   do_segments=False)

    with pts.phase('quality computation'):
        predicted_segment_list_rectified = predict_segments(sm=assumed_axle, gpg=gpg)
        quality_res, quality_stats = judge_quality(image, segment_list2_rect, predicted_segment_list_rectified)
        res.update(quality_res)

    #     res['blurred']= cv2.medianBlur(image, 11)
    stats['estimate'] = est
    stats.update(quality_stats)

    dtu.logger.info(pts.get_stats())
    return res, stats


def judge_quality(image, observed_segment_list, predicted_segment_list):
    res = OrderedDict()
    stats = OrderedDict()

    #     mask2gray = lambda x: (x * 255).clip(0, 255).astype('uint8')
    H, W, _ = image.shape
    r = 1
    reason_shape = (H * r, W * r, 3)
    observed_width = int(14 * r)
    predicted_width = int(50 * r)

    summary_bgr = np.zeros(reason_shape, 'uint8')

    ratios = []
    for color in [Segment.WHITE, Segment.YELLOW, Segment.RED, ]:
        predicted = only_one_color(predicted_segment_list, color)
        predicted_mask = np.zeros(reason_shape[0:2], 'float32')
        _draw_segment_list_on_image(predicted_mask, segment_list=predicted, width=predicted_width)

        predicted_mask = np.sign(predicted_mask) * 255

        observed = only_one_color(observed_segment_list, color)
        observed_mask = np.zeros(reason_shape[0:2], 'float32')
        _draw_segment_list_on_image(observed_mask, segment_list=observed, width=observed_width)
        observed_mask = np.sign(observed_mask)

        explained = np.sign(observed_mask * predicted_mask)
        not_explained = observed_mask - explained

        _B, G, R = 0, 1, 2
        summary_bgr[:, :, R] |= (not_explained * 255).astype('uint8')
        summary_bgr[:, :, G] |= (explained * 255).astype('uint8')

        ratio_explained = 1 - (np.sum(not_explained) / (np.sum(explained) + np.sum(not_explained) + 1))
        ratios.append(ratio_explained)
    #         res['predicted %s' % color] = mask2gray(predicted_mask)
    #         res['observed %s' % color] = mask2gray(observed_mask)
    #         res['explained %s = %d%%' % (color, ratio_explained*100)] = mask2gray(explained)

    avg = np.mean(ratios)

    percent = lambda x: '%d%%' % (x * 100)
    s = "W %s Y %s R %s" % (percent(ratios[0]), percent(ratios[1]), percent(ratios[2]))
    res['explained %s [%s]' % (percent(avg), s)] = summary_bgr

    stats['quality'] = avg
    stats['ratios'] = ratios
    return res, stats


def _draw_segment_list_on_image(mask, segment_list, width):
    shape = mask.shape[:2]

    for segment in segment_list.segments:
        p1 = segment.pixels_normalized[0]
        p2 = segment.pixels_normalized[1]

        P1 = normalized_to_image(p1, shape)
        P2 = normalized_to_image(p2, shape)

        paint = 255

        cv2.line(mask, P1, P2, paint, width)


@dtu.contract(segment_list=SegmentList, color=int, returns=SegmentList)
def only_one_color(segment_list, color):
    segments = [s for s in segment_list.segments if s.color == color]
    return SegmentList(segments=segments)


def get_grid(shape, L=32, col={0: (255, 0, 0), 1: (0, 255, 0)}):
    """ Creates a grid of given shape """
    H, W = shape
    res = np.zeros((H, W, 3), 'uint8')
    for i in range(H):
        for j in range(W):
            cx = int(i / L)
            cy = int(j / L)
            coli = (cx + cy) % 2
            res[i, j, :] = col[coli]
    return res

def printWtime(msg):
    print("["+datetime.now().strftime("%x @ %X.%f")+"] "+msg)
from pose_format import PoseHeader
from pose_format.pose_header import PoseHeaderDimensions


def get_pose_header():
    from pose_format.utils.holistic import holistic_components

    pose_components = [c for c in holistic_components()
                       if c.name in ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]]
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    return PoseHeader(version=0.1, dimensions=dimensions, components=pose_components)


if __name__ == "__main__":
    pose_header = get_pose_header()
    with open("holistic.header", "wb") as f:
        pose_header.write(f)

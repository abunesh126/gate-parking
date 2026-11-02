"""Independent Windows camera enumeration and frame-access diagnostic."""

import cv2

try:
    from cv2_enumerate_cameras import enumerate_cameras, supported_backends
except ImportError:
    enumerate_cameras = None
    supported_backends = ()


def main() -> None:
    if enumerate_cameras is not None:
        print("Named Windows devices:")
        for backend in supported_backends:
            print(f"\nBackend constant: {backend}")
            for device in enumerate_cameras(backend):
                print(
                    f"  name={device.name!r} index={device.index} "
                    f"path={device.path!r}".encode("ascii", "backslashreplace").decode()
                )
    else:
        print("cv2-enumerate-cameras is not installed; using index fallback.")

    print("\nValidated OpenCV index access:")
    for backend_name, backend in (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("OpenCV default", cv2.CAP_ANY),
    ):
        for index in range(20):
            capture = cv2.VideoCapture(index, backend)
            try:
                ok, frame = capture.read() if capture.isOpened() else (False, None)
                if ok and frame is not None and frame.size:
                    height, width = frame.shape[:2]
                    print(f"  index={index} backend={backend_name} resolution={width}x{height}")
            finally:
                capture.release()


if __name__ == "__main__":
    main()

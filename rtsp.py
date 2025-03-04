import cv2
import urllib3

def read_rtsp_stream(rtsp_url):
    """
    Reads an RTSP stream and displays the frames.

    Args:
        rtsp_url: The URL of the RTSP stream.
    """

    # Suppress urllib3 warnings about insecure connections (if necessary)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        # Open the RTSP stream
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return

        while True:
            # Read a frame from the stream
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the frame (optional - uncomment to show)
            cv2.imshow('RTSP Stream', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    rtsp_url = "rtsp://ailab:adani123@192.168.1.64:554/Streaming/Channels/1"  # Replace with your RTSP URL
    read_rtsp_stream(rtsp_url)

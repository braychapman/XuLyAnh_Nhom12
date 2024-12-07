import cv2
import mediapipe as mp
import time

# Lớp handDetector dùng để phát hiện và xử lý dữ liệu bàn tay từ MediaPipe
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Khởi tạo đối tượng handDetector.

        Args:
            mode (bool): Chế độ tĩnh (static_image_mode). True nếu chỉ xử lý ảnh tĩnh, False nếu xử lý video.
            maxHands (int): Số lượng bàn tay tối đa cần phát hiện.
            detectionCon (float): Ngưỡng độ tin cậy cho việc phát hiện bàn tay (min_detection_confidence).
            trackCon (float): Ngưỡng độ tin cậy cho việc theo dõi bàn tay (min_tracking_confidence).
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        # Khởi tạo MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,  # Chế độ tĩnh hoặc động
            max_num_hands=self.maxHands, # Số lượng bàn tay tối đa
            min_detection_confidence=self.detectionCon, # Độ tin cậy khi phát hiện
            min_tracking_confidence=self.trackCon # Độ tin cậy khi theo dõi
        )
        self.mpDraw = mp.solutions.drawing_utils # Dùng để vẽ landmarks

    def findHands(self, img, draw=True):
        """
        Phát hiện bàn tay trong ảnh.

        Args:
            img (numpy.ndarray): Ảnh đầu vào.
            draw (bool): Vẽ landmarks lên ảnh nếu True.

        Returns:
            numpy.ndarray: Ảnh đầu ra với landmarks (nếu draw=True).
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển đổi ảnh sang RGB
        self.results = self.hands.process(imgRGB) # Xử lý ảnh để phát hiện bàn tay

        if self.results.multi_hand_landmarks: # Kiểm tra nếu có tìm thấy bàn tay
            for handLms in self.results.multi_hand_landmarks: # Duyệt qua các landmarks của mỗi bàn tay
                if draw:
                    self.mpDraw.draw_landmarks( # Vẽ landmarks lên ảnh
                        img, handLms, self.mpHands.HAND_CONNECTIONS) # Sử dụng HAND_CONNECTIONS để nối các điểm
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Tìm tọa độ của các landmarks bàn tay.

        Args:
            img (numpy.ndarray): Ảnh đầu vào.
            handNo (int): Số thứ tự của bàn tay (0 là bàn tay đầu tiên được phát hiện).
            draw (bool): Vẽ các điểm landmarks lên ảnh nếu True.

        Returns:
            list: Danh sách các landmarks với định dạng [id, cx, cy],  trong đó cx, cy là tọa độ x, y. Trả về list rỗng nếu không tìm thấy tay.
        """
        lmList = []
        if self.results.multi_hand_landmarks: # Kiểm tra nếu có tìm thấy bàn tay
            myHand = self.results.multi_hand_landmarks[handNo] # Lấy landmarks của bàn tay thứ handNo
            for id, lm in enumerate(myHand.landmark): # Duyệt qua các landmarks
                h, w, c = img.shape # Lấy chiều cao, chiều rộng và số kênh của ảnh
                cx, cy = int(lm.x * w), int(lm.y * h) # Tính toán tọa độ x, y trên ảnh
                lmList.append([id, cx, cy]) # Thêm vào danh sách
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED) # Vẽ điểm
        return lmList

def main():
    pTime = 0 # Thời gian trước đó
    cap = cv2.VideoCapture(0) # Khởi tạo camera
    detector = handDetector() # Khởi tạo đối tượng handDetector

    while True:
        success, img = cap.read() # Đọc ảnh từ camera
        if not success: # Kiểm tra xem có đọc được ảnh hay không
            break

        try:
            img = detector.findHands(img) # Phát hiện bàn tay
            lmList = detector.findPosition(img) # Tìm tọa độ landmarks

            if len(lmList) != 0: # Nếu tìm thấy bàn tay
                print(lmList[4]) # In ra tọa độ landmarks của điểm thứ 4 (ví dụ)

        except Exception as e:
            print(f"Lỗi xảy ra trong xử lý mediapipe: {e}") # Báo lỗi nếu có lỗi xảy ra

        cTime = time.time() # Thời gian hiện tại
        fps = 1 / (cTime - pTime) # Tính toán FPS
        pTime = cTime # Cập nhật thời gian trước đó
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, # Hiển thị FPS lên ảnh
                    (255, 0, 255), 3)

        cv2.imshow("Image", img) # Hiển thị ảnh
        if cv2.waitKey(1) == ord("q"): # Thoát chương trình nếu nhấn q
            break

    cap.release() # Giải phóng camera
    cv2.destroyAllWindows() # Đóng tất cả cửa sổ


if __name__ == "__main__":
    main()
import tkinter as tk
import cv2
import hand as htm
import threading
from PIL import Image, ImageTk

class HandDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Phát hiện bàn tay")

        self.detector = htm.handDetector(detectionCon=0.75, maxHands=2)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_label = tk.Label(master, text="Không thể mở camera!", fg="red")
            self.error_label.pack(pady=20)
            return #Thoát nếu không mở được camera

        self.running = False

        self.label = tk.Label(master)
        self.label.pack()

        self.start_button = tk.Button(master, text="Bắt đầu", command=self.start)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Dừng", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.finger_count_label = tk.Label(master, text="Số ngón tay: 0")
        self.finger_count_label.pack(pady=10)


    def start(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def process_video(self):
        fingerid = [4, 8, 12, 16, 20]

        while self.running:
            success, frame = self.cap.read()
            if not success:
                break

            frame = self.detector.findHands(frame, draw=True)
            lmLists = self.detector.results.multi_hand_landmarks

            total_fingers = 0
            if lmLists:
                for hand_no, handLms in enumerate(lmLists):
                    lmList = self.detector.findPosition(frame, handNo=hand_no, draw=False)
                    fingers = []
                    

                    # Kiểm tra ngón cái
                    if lmList[17][1] > lmList[5][1]:  # Tay phải
                        if lmList[fingerid[0]][1] < lmList[fingerid[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:  # Tay trái
                        if lmList[fingerid[0]][1] > lmList[fingerid[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Kiểm tra các ngón tay khác
                    for id in range(1, 5):
                        if lmList[fingerid[id]][2] < lmList[fingerid[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    total_fingers += fingers.count(1)


            cv2.putText(frame, f"Total Fingers: {total_fingers}", (10, 400),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)
            self.finger_count_label.config(text=f"Số ngón tay: {total_fingers}")
            self.master.update()

        self.cap.release() # Giải phóng camera khi dừng


root = tk.Tk()
app = HandDetectionApp(root)
root.mainloop()
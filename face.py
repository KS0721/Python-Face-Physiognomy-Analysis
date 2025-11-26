import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk

# --- 1. 관상 분석 로직 및 특징점 정의 ---

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
# static_image_mode=False: 비디오 스트림에 최적화
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 관상 분석에 사용할 주요 특징점 인덱스 (MediaPipe 468개 기준)
LANDMARK_INDICES = {
    # 초년운 (눈 간 거리 기준)
    'LEFT_EYE_INNER': 362,
    'RIGHT_EYE_INNER': 133,
    
    # 중년운 (코 너비 기준)
    'NOSE_LEFT_FLANK': 142,
    'NOSE_RIGHT_FLANK': 371,
    
    # 말년운 (턱 끝 길이 기준)
    'CHIN_TIP': 152,  
    'NOSE_TIP': 1,    
    
    # 얼굴 전체 너비 기준점
    'LEFT_CHEEK': 234,
    'RIGHT_CHEEK': 454
}

def get_distance(landmarks, p1_idx, p2_idx):
    """두 특징점 간의 유클리디안 거리를 계산합니다."""
    try:
        # 특징점 좌표 추출 (0.0에서 1.0 사이의 상대 좌표)
        p1 = np.array([landmarks[p1_idx].x, landmarks[p1_idx].y])
        p2 = np.array([landmarks[p2_idx].x, landmarks[p2_idx].y])
        return hypot(p1[0] - p2[0], p1[1] - p2[1])
    except IndexError:
        return 0

def analyze_physiognomy(landmarks):
    """추출된 특징점을 바탕으로 초년, 중년, 말년 관상 분석을 수행합니다."""
    
    analysis = {}
    
    # 얼굴 비율 계산에 사용될 주요 값
    face_width = get_distance(landmarks, LANDMARK_INDICES['LEFT_CHEEK'], LANDMARK_INDICES['RIGHT_CHEEK'])
    eye_inner_distance = get_distance(landmarks, LANDMARK_INDICES['LEFT_EYE_INNER'], LANDMARK_INDICES['RIGHT_EYE_INNER'])
    nose_width = get_distance(landmarks, LANDMARK_INDICES['NOSE_LEFT_FLANK'], LANDMARK_INDICES['NOSE_RIGHT_FLANK'])
    
    # 예외 처리: 얼굴 너비가 0이면 비율 계산 불가
    nose_to_face_ratio = nose_width / face_width if face_width > 0 else 0

    # 코와 턱의 세로 길이 (하관의 길이)
    lower_face_length = 0
    try:
        nose_y = landmarks[LANDMARK_INDICES['NOSE_TIP']].y
        chin_y = landmarks[LANDMARK_INDICES['CHIN_TIP']].y
        lower_face_length = chin_y - nose_y
    except:
        pass 

    # --- 1. 초년운 (Early Fortune: 이마/눈) ---
    # 눈 간 거리가 얼굴 너비의 15% 이상일 경우
    if eye_inner_distance > (face_width * 0.15): 
        analysis['early_fortune'] = {
            'title': "🥇 초년운 (이마/눈)",
            'feature': "눈 간 거리가 적당하여 시야가 넓고 포용력이 좋습니다.",
            'fortune': "초년운이 순탄하고 대인 관계에서 복을 얻습니다."
        }
    else:
        analysis['early_fortune'] = {
            'title': "🥇 초년운 (이마/눈)",
            'feature': "눈 간 거리가 좁은 편이어서 집중력과 몰입도가 뛰어납니다.",
            'fortune': "학업 및 한 분야에 재능을 발휘하며 목표를 향한 집념이 강합니다."
        }
        
    # --- 2. 중년운 (Middle Fortune: 코/재물) ---
    # 코 너비가 얼굴 너비의 10% 이상일 경우
    if nose_to_face_ratio > 0.1: 
        analysis['middle_fortune'] = {
            'title': "💵 중년운 (코/재물)",
            'feature': f"코의 폭이 넓고 콧방울이 두툼하여 재물을 담는 그릇이 큽니다. (비율: {nose_to_face_ratio:.2f})",
            'fortune': "재물운이 강하며, 중년 이후 부를 축적할 가능성이 높습니다."
        }
    else:
        analysis['middle_fortune'] = {
            'title': "💵 중년운 (코/재물)",
            'feature': f"코가 오똑하고 날렵하여 명예를 중시하는 관상입니다. (비율: {nose_to_face_ratio:.2f})",
            'fortune': "직업운과 명예운이 좋으며, 꾸준한 노력으로 재물을 모읍니다."
        }
        
    # --- 3. 말년운 (Later Fortune: 턱/하관) ---
    # 하관(코 끝~턱 끝)의 길이가 충분하면 말년이 안정적이라고 해석
    if lower_face_length > 0.35: 
        analysis['later_fortune'] = {
            # "후년운"을 "말년운"으로 명확히 수정
            'title': "👵 말년운 (턱/하관)",
            'feature': "턱선이 발달하고 하관이 길어 튼튼하고 안정적인 인상입니다.",
            'fortune': "말년운과 건강운이 좋습니다. 자손과의 관계도 원만하여 평안합니다."
        }
    else:
        analysis['later_fortune'] = {
            # "후년운"을 "말년운"으로 명확히 수정
            'title': "👵 말년운 (턱/하관)",
            'feature': "턱이 짧거나 좁은 편으로 활동적이고 민첩한 인상입니다.",
            'fortune': "말년의 복을 위해 꾸준한 건강 관리와 여가 활동 준비가 중요합니다."
        }

    return analysis


# --- 2. GUI 및 카메라 통합 로직 (tkinter / OpenCV) ---

class FaceAnalysisApp:
    def __init__(self, window, window_title="Gemini 관상 분석기"):
        self.window = window
        self.window.title(window_title)

        # 카메라 캡처 객체 초기화 (0번 카메라)
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
             messagebox.showerror("오류", "카메라를 찾을 수 없습니다. (카메라 연결 또는 권한 확인 필요)")
             self.window.destroy()
             return

        # 비디오 출력 프레임 크기 설정
        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. 비디오 디스플레이 영역
        self.canvas = tk.Canvas(window, width=width, height=height, bg="black")
        self.canvas.pack(padx=10, pady=10)

        # 2. 캡처 버튼
        self.btn_capture=tk.Button(window, text="관상 분석 시작", width=50, command=self.capture_and_analyze, 
                                   font=("맑은 고딕", 12, "bold"), fg="white", bg="#4A90E2")
        self.btn_capture.pack(anchor=tk.CENTER, expand=True, pady=(0, 10))

        # 3. 분석 결과 표시 영역 (스크롤 가능 텍스트 박스)
        self.result_label = scrolledtext.ScrolledText(window, height=10, width=80, 
                                                     wrap=tk.WORD, font=("맑은 고딕", 10))
        self.result_label.insert(tk.END, "얼굴을 정면에 맞추고 버튼을 눌러주세요. (초년운, 중년운, 말년운 분석)")
        self.result_label.config(state=tk.DISABLED) # 읽기 전용으로 설정
        self.result_label.pack(anchor=tk.W, fill=tk.X, padx=10, pady=(0, 10))

        # 15ms마다 화면 업데이트
        self.delay = 15
        self.update()

        # 창 닫기 이벤트에 카메라 해제 함수 연결
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        """카메라 프레임을 읽어와 GUI에 표시합니다."""
        ret, frame = self.vid.read()

        if ret:
            # 좌우 반전 및 색상 변환 (사용자에게 거울처럼 보이도록)
            frame = cv2.flip(frame, 1) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

        self.window.after(self.delay, self.update)

    def capture_and_analyze(self):
        """현재 프레임을 캡처하여 관상 분석을 수행하고 결과를 표시합니다."""
        ret, frame = self.vid.read()
        if not ret:
            self.update_result_text("⚠️ 카메라 프레임 캡처에 실패했습니다.")
            return
        
        # 분석을 위해 좌우 반전 및 RGB 변환
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 특징점 감지
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.update_result_text("⚠️ 얼굴이 감지되지 않았습니다. 정면을 바라보고 다시 시도해 주세요.")
            return
        
        # 특징점 추출 및 분석 실행
        landmarks = results.multi_face_landmarks[0].landmark
        analysis = analyze_physiognomy(landmarks)

        # 결과 텍스트 포맷팅
        result_text = "=================================\n"
        result_text += "⭐ Gemini 관상 분석 완료 (초/중/말년) ⭐\n"
        result_text += "=================================\n"
        
        for key, item in analysis.items():
            result_text += f"\n[ {item['title']} ]\n"
            result_text += f" - 특징: {item['feature']}\n"
            result_text += f" - 해설: {item['fortune']}\n"

        self.update_result_text(result_text)

    def update_result_text(self, text):
        """결과 텍스트 박스의 내용을 업데이트합니다."""
        self.result_label.config(state=tk.NORMAL)
        self.result_label.delete('1.0', tk.END)
        self.result_label.insert(tk.END, text)
        self.result_label.config(state=tk.DISABLED)

    def on_closing(self):
        """창 닫기 이벤트 처리: 카메라 리소스 해제"""
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

# --- 3. 애플리케이션 실행 ---

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)
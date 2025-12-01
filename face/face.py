import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk 
from PIL import Image, ImageTk

# --- MediaPipe 및 Drawing Utility 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# --- 1. 관상 분석에 사용될 주요 특징점 인덱스 (MediaPipe 468개 기준) ---

LANDMARK_INDICES = {
    # --- 삼정 분석을 위한 세로 분할 기준점 ---
    'HAIRLINE_CENTER': 10,     
    'BROW_CENTER': 9,          
    'NOSE_TIP': 1,             
    'CHIN_TIP': 152,           
    
    # --- 오악 (3D 돌출도) 분석 기준점 ---
    'FOREHEAD_CENTER': 10,     
    'NOSE_BRIDGE': 6,          
    'CHIN_PROJECTION': 152,    
    'LEFT_CHEEK_MAX': 234,     
    'RIGHT_CHEEK_MAX': 454,
    
    # --- 안색 분석을 위한 영역 중심점 (코 주변) ---
    'CHEEK_TONE_SAMPLE': 454,  
    
    # --- 초년/중년/말년 기본 분석 기준점 ---
    'LEFT_EYE_INNER': 362,
    'RIGHT_EYE_INNER': 133,
    'NOSE_LEFT_FLANK': 142,
    'NOSE_RIGHT_FLANK': 371,
    'LEFT_MOUTH_CORNER': 61,
    'RIGHT_MOUTH_CORNER': 291
}

def get_distance(landmarks, p1_idx, p2_idx):
    """두 특징점 간의 유클리디안 거리를 계산합니다."""
    try:
        p1 = np.array([landmarks[p1_idx].x, landmarks[p1_idx].y])
        p2 = np.array([landmarks[p2_idx].x, landmarks[p2_idx].y])
        return hypot(p1[0] - p2[0], p1[1] - p2[1])
    except IndexError:
        return 0

# --- 2. 전문가 관상 분석 로직 함수 ---

def analyze_physiognomy(landmarks, frame):
    """
    추출된 특징점과 프레임 이미지(안색 분석용)를 바탕으로
    삼정, 오악, 안색 등 전문적인 관상 분석을 수행합니다.
    (모든 항목에 4단계 값 폭 분할 및 길흉 통합 해설 적용)
    """
    
    analysis = {}
    H, W, _ = frame.shape
    
    # --- 기본 비율 측정 ---
    face_width = get_distance(landmarks, LANDMARK_INDICES['LEFT_CHEEK_MAX'], LANDMARK_INDICES['RIGHT_CHEEK_MAX'])
    eye_inner_distance = get_distance(landmarks, LANDMARK_INDICES['LEFT_EYE_INNER'], LANDMARK_INDICES['RIGHT_EYE_INNER'])
    nose_width = get_distance(landmarks, LANDMARK_INDICES['NOSE_LEFT_FLANK'], LANDMARK_INDICES['NOSE_RIGHT_FLANK'])
    nose_to_face_ratio = nose_width / face_width if face_width > 0 else 0
    
    lower_face_length = 0
    try:
        nose_y = landmarks[LANDMARK_INDICES['NOSE_TIP']].y
        chin_y = landmarks[LANDMARK_INDICES['CHIN_TIP']].y
        lower_face_length = chin_y - nose_y
    except:
        pass 

    # -------------------------------------------------------------
    # A. 3D 돌출도 분석 (오악의 입체감 - 중앙악) - **4단계 분할 적용**
    # -------------------------------------------------------------
    try:
        forehead_z = landmarks[LANDMARK_INDICES['FOREHEAD_CENTER']].z
        nose_z = landmarks[LANDMARK_INDICES['NOSE_BRIDGE']].z
        nose_prominence_score = forehead_z - nose_z
        
        if nose_prominence_score > 0.07: # 매우 돌출
            analysis['prominence'] = {
                'title': "🏔️ 3D 입체감 (중앙악)",
                'feature': f"콧대가 **매우 뚜렷하게** 돌출되어 의지력이 극히 강합니다. (점수: {nose_prominence_score:.3f})",
                'fortune': "타인을 압도하는 **강한 리더십과 독립심**을 가집니다. **과감한 투쟁 정신**으로 큰 성과를 거두지만, 지나친 자신감과 **오만**으로 인해 인간관계에서 고립되거나 **주변의 반발**을 살 위험이 매우 높습니다."
            }
        elif nose_prominence_score > 0.03: # 적당히 돌출
            analysis['prominence'] = {
                'title': "🏔️ 3D 입체감 (중앙악)",
                'feature': f"콧대가 뚜렷하여 의지력이 강합니다. (점수: {nose_prominence_score:.3f})",
                'fortune': "자신의 분야에서 **확고한 주체성**을 가지고 목표를 개척합니다. **추진력**이 강하지만, 가끔 타인의 의견을 무시하고 **독단적으로 행동**하는 경향이 있으니 유연성을 길러야 합니다."
            }
        elif nose_prominence_score < 0.01: # 매우 평평
            analysis['prominence'] = {
                'title': "🏔️ 3D 입체감 (중앙악)",
                'feature': f"콧대가 매우 평평하여 윤곽이 희미합니다. (점수: {nose_prominence_score:.3f})",
                'fortune': "성격이 **매우 온순**하고 주변과 갈등을 피하지만, **주체성이 극히 약해** 중요한 결정에서 **쉽게 흔들리거나** **소극적**으로 임할 수 있습니다. 자신의 목소리를 낼 필요가 있습니다."
            }
        else: # 조화로운 경우
            analysis['prominence'] = {
                'title': "🏔️ 3D 입체감 (중앙악)",
                'feature': f"콧대가 얼굴의 전체 윤곽과 조화롭게 어우러집니다. (점수: {nose_prominence_score:.3f})",
                'fortune': "타인과의 **조화와 협력을 중시**하며, 어떤 조직에서도 융화력이 뛰어납니다. 다만, **자신의 주장을 강하게 펼치지 못해** 중요한 기회를 놓치거나, **주변 환경에 쉽게 휘둘릴** 수 있으니 주체성을 강화해야 합니다."
            }
    except Exception:
        pass


    # -------------------------------------------------------------
    # B. 삼정(三停) 균형 분석 (인생의 단계별 안정성) - **4단계 분할 적용**
    # -------------------------------------------------------------
    try:
        upper_y = landmarks[LANDMARK_INDICES['HAIRLINE_CENTER']].y
        middle_y = landmarks[LANDMARK_INDICES['BROW_CENTER']].y
        nose_y_삼정 = landmarks[LANDMARK_INDICES['NOSE_TIP']].y
        chin_y_삼정 = landmarks[LANDMARK_INDICES['CHIN_TIP']].y
        
        upper_length = middle_y - upper_y
        middle_length = nose_y_삼정 - middle_y
        lower_length_삼정 = chin_y_삼정 - nose_y_삼정
        total_length = upper_length + middle_length + lower_length_삼정
        
        ratio_upper = upper_length / total_length
        ratio_middle = middle_length / total_length
        ratio_lower = lower_length_삼정 / total_length
        
        fortunes = []
        
        # 1. 상정 (초년운) 분석
        if ratio_upper < 0.28: # 매우 짧음
            fortunes.append("상정이 **매우 짧아** 초년(30세 이전)에 **극도로 자수성가**해야 했으며, **부모나 환경의 도움을 기대하기 어렵습니다.** 오직 실행력으로 승부해야 합니다.")
        elif ratio_upper < 0.32: # 약간 짧음
            fortunes.append("상정이 짧은 편이라 초년에 **자수성가형** 기질이 강합니다. 지적 능력보다는 **현실적인 성과**를 빨리 내는 것에 집중합니다.")
        elif ratio_upper > 0.38: # 매우 긺
            fortunes.append("상정이 **매우 길어** 초년운이 순탄하고 **학문, 명예, 부모의 덕**이 두텁습니다. 다만, **현실 감각이 매우 부족**하거나 **이상주의에 빠져** 사회 적응이 늦어질 수 있습니다.")
        elif ratio_upper > 0.34: # 약간 긺
            fortunes.append("상정이 긴 편이라 초년에 **지적인 성장**과 환경적인 혜택을 잘 받습니다. **논리적인 사고**가 강점이나, 때로 **융통성이 부족**해 보일 수 있습니다.")
        else:
            fortunes.append("상정이 균형적으로 적당하여 초년의 성장이 안정적이며, **인생의 시작이 순조롭습니다.**")

        # 2. 중정 (중년운) 분석
        if ratio_middle < 0.28: # 매우 짧음
            fortunes.append("중정이 **매우 짧아** 직업 운세가 불안정할 수 있습니다. **책임감이 부족**해 보이거나 **일관성이 매우 없는** 사람으로 비쳐 신뢰를 잃지 않도록 주의해야 합니다.")
        elif ratio_middle < 0.32: # 약간 짧음
            fortunes.append("중정이 짧아 명예보다는 실속을 중시하며, **직업 변동**이 있을 수 있습니다. **재테크에 신중함**이 필요합니다.")
        elif ratio_middle > 0.38: # 매우 긺
            fortunes.append("중정이 **매우 길어** **중년(31~50세)의 운세가 압도적으로 왕성**합니다. 재물운, 명예, 사업운이 크게 발달하나, **지나친 탐욕**으로 인해 주변 사람들을 잃거나 큰 구설수에 오를 수 있습니다.")
        elif ratio_middle > 0.34: # 약간 긺
            fortunes.append("중정이 긴 편이라 중년의 **재물운과 사회적 성취**가 좋습니다. 꾸준한 노력으로 큰 결실을 맺으나, **명예욕**이 과해지지 않도록 주의해야 합니다.")
        else:
            fortunes.append("중정이 균형적이어서 중년의 직업과 사회 활동이 **안정적**이며, 노력한 만큼의 결실을 얻습니다.")

        # 3. 하정 (말년운) 분석
        if ratio_lower < 0.28: # 매우 짧음
            fortunes.append("하정이 **매우 짧아** **인내심과 지구력이 극히 부족**합니다. 말년에 **건강과 인복**이 약해져 **매우 고독**해지지 않도록 젊을 때부터 장기적인 계획과 건강 관리가 필수입니다.")
        elif ratio_lower < 0.32: # 약간 짧음
            fortunes.append("하정이 짧아 **활동적**이지만 **지구력이 약한 편**입니다. 말년의 복을 위해 꾸준한 건강 관리와 재정적 안정이 중요하며, **인복을 쌓는 데 집중**해야 합니다.")
        elif ratio_lower > 0.38: # 매우 긺
            fortunes.append("하정이 **매우 길고** 튼튼하여 **말년(51세 이후)의 인복과 건강운이 최고**입니다. 다만, **고집이 너무 세서** 주변의 변화를 전혀 수용하지 않아 **독선적**이라는 평을 들을 수 있습니다.")
        elif ratio_lower > 0.34: # 약간 긺
            fortunes.append("하정이 긴 편이라 **말년운이 안정적**입니다. **인내심과 지구력**이 강해 중년의 성과를 잘 지켜냅니다. 다만, **보수적인 성향**이 강해 새로운 시도를 꺼릴 수 있습니다.")
        else:
            fortunes.append("하정이 균형적이어서 말년의 생활이 **평화롭고 안정적**이며, 주변과의 관계가 원만합니다.")

        final_fortune = " ".join(fortunes)

        analysis['three_stops'] = {
            'title': "⚖️ 삼정(三停) 균형 분석",
            'feature': f"상정: {ratio_upper:.2f}, 중정: {ratio_middle:.2f}, 하정: {ratio_lower:.2f}",
            'fortune': final_fortune
        }
    except Exception:
        pass


    # -------------------------------------------------------------
    # C. 안색/기색 분석 (활력/건강) - **4단계 분할 및 디테일 강화**
    # -------------------------------------------------------------
    try:
        lm = landmarks[LANDMARK_INDICES['CHEEK_TONE_SAMPLE']]
        x_px = int(lm.x * W)
        y_px = int(lm.y * H)
        
        sample_area = frame[max(0, y_px-5):min(H, y_px+5), max(0, x_px-5):min(W, x_px+5)]
        avg_color = np.mean(sample_area, axis=(0, 1))
        R, G, B = avg_color
        avg_brightness = (R + G + B) / 3
        
        if R > G + 15 and R > B + 15: # 붉은 기운이 매우 강한 경우
            tone_msg = "**양기(陽氣)가 지나치게 왕성**하여 매사에 자신감이 넘치지만, **과도한 스트레스**나 **분노**가 잠재되어 있습니다. 즉시 휴식을 취하지 않으면 **건강 문제(화병)**를 초래할 수 있습니다."
        elif avg_brightness < 70 and G < 70: # 매우 어둡고 푸른 기운이 도는 경우
             tone_msg = "안색이 **매우 어둡고 침체된 기운**을 띄어 **심각한 피로**가 누적되었거나, **금전적으로 큰 고민**을 안고 있을 수 있습니다. 운기가 최저조일 수 있으니 모든 중요한 결정을 연기해야 합니다."
        elif avg_brightness < 90: # 전반적으로 어두운 경우
             tone_msg = "안색이 약간 어두운 기운을 띄어 **피로도가 누적**되었거나 **심리적으로 고민**이 있습니다. 건강 관리에 주의하고 중요한 결정 전 심신의 안정이 필요합니다."
        elif avg_brightness > 140: # 전반적으로 밝은 경우
             tone_msg = "안색이 **매우 밝고 윤기가 흘러** 현재 **운기가 최고조**입니다. 마음이 평온하고 활력이 넘치며, 모든 일이 순조롭게 진행될 징조입니다."
        else:
            tone_msg = "안색이 **평온하고 맑은 기운**을 유지하고 있습니다. **마음이 안정**되어 있고, 현재의 노력이 곧 **인복과 재물운**으로 이어질 준비가 된 상태입니다."
            
        analysis['tone'] = {
            'title': "☯️ 안색/기색 분석",
            'feature': f"평균 RGB: ({R:.0f}, {G:.0f}, {B:.0f}), 밝기: {avg_brightness:.0f}",
            'fortune': tone_msg
        }
    except Exception:
        pass
        
    # -------------------------------------------------------------
    # D. 초년, 중년, 말년 기본 운세 분석 - **맞춤형 수치 기반 해설**
    # -------------------------------------------------------------
    
    # --- 1. 초년운 (이마/눈 - 눈 간 거리) ---
    eye_ratio_threshold = face_width * 0.15 
    
    if eye_inner_distance > eye_ratio_threshold * 1.15: # 매우 넓은 경우
        analysis['early_fortune'] = {
            'title': "🥇 초년운 (이마/눈)",
            'feature': f"눈 간 거리가 **매우 넓어** 시야가 광활하고 포용력이 뛰어납니다. (비율: {eye_inner_distance/face_width:.3f})",
            'fortune': "세상을 넓게 보는 **뛰어난 안목**과 포용력으로 인복을 잘 받습니다. 하지만 **집중력이 매우 부족**하거나 **결단력이 약해** 우유부단하다는 평을 듣기 쉬우니, 목표 설정에 집중해야 합니다."
        }
    elif eye_inner_distance > eye_ratio_threshold * 0.9: # 적당한 경우
        analysis['early_fortune'] = {
            'title': "🥇 초년운 (이마/눈)",
            'feature': f"눈 간 거리가 적당하여 시야가 넓고 포용력이 좋습니다. (비율: {eye_inner_distance/face_width:.3f})",
            'fortune': "초년운이 순탄하여 주변의 **지적인 도움이나 환경적인 혜택**을 잘 받습니다. **포용력**이 강해 다양한 사람과 긍정적인 관계를 맺습니다."
        }
    else: # 좁은 경우
        analysis['early_fortune'] = {
            'title': "🥇 초년운 (이마/눈)",
            'feature': f"눈 간 거리가 좁은 편이어서 집중력과 몰입도가 뛰어납니다. (비율: {eye_inner_distance/face_width:.3f})",
            'fortune': "학업 및 한 분야에 **집중력과 몰입도**가 뛰어나 전문 분야에서 큰 성과를 보지만, 시야가 좁아 **세부 사항에 매몰**되거나, **융통성이 부족**하여 대인 관계에서 오해를 살 수 있으니 폭넓은 사고가 필요합니다."
        }
    
    # --- 2. 중년운 (코/재물) ---
    if nose_to_face_ratio > 0.12: # 매우 넓은 경우
        strength = "**압도적으로 강합니다**"
        risk = "지나친 탐욕과 과소비"
        
        analysis['middle_fortune'] = {
            'title': "💵 중년운 (코/재물)",
            'feature': f"코의 폭이 넓고 콧방울이 두툼하여 재물을 담는 그릇이 큽니다. (비율: {nose_to_face_ratio:.2f})",
            'fortune': f"재물운이 **{strength}** 축재 능력이 뛰어납니다. 중년에 큰 사업적 성과를 거두지만, **{risk}**이 과해져 주변 사람들에게 인색하게 굴거나 도덕적인 문제에 휘말릴 위험이 높습니다. 재물을 **의롭게** 쓰는 것이 중요합니다."
        }
    elif nose_to_face_ratio > 0.09: # 적당히 넓은 경우
        strength = "상당히 강합니다"
        risk = "재물에 대한 집착"
        
        analysis['middle_fortune'] = {
            'title': "💵 중년운 (코/재물)",
            'feature': f"코의 폭이 넓어 재물을 담는 그릇이 큽니다. (비율: {nose_to_face_ratio:.2f})",
            'fortune': f"재물운이 **{strength}** 중년 이후 부를 축적할 가능성이 높습니다. 하지만 **{risk}**이 강해져 주변과 금전적인 마찰을 겪거나, 공적인 명예를 잃을 위험이 있습니다."
        }
    else: # 좁거나 날렵한 경우
        strength = "좋으며 명예를 중시합니다" if nose_to_face_ratio > 0.07 else "뛰어나지만 실속이 약합니다"
        risk = "자존심이 강해 도움을 거부"
        
        analysis['middle_fortune'] = {
            'title': "💵 중년운 (코/재물)",
            'feature': f"코가 오똑하고 날렵하여 명예를 중시하는 관상입니다. (비율: {nose_to_face_ratio:.2f})",
            'fortune': f"직업운과 명예운이 **{strength}**. 지위와 명성을 통해 재물을 부릅니다. 다만, **재물 관리가 서툴러** 큰돈을 모으기 어려울 수 있으며, **{risk}**하여 기회를 놓치기도 합니다. 비율({nose_to_face_ratio:.2f})이 낮을수록 명예를 더 중시합니다."
        }
    
    # --- 3. 말년운 (턱/하관) ---
    if lower_face_length > 0.4: # 매우 긴 경우
        strength = "**최고 수준의 안정성**"
        risk = "**극도의 독선**과 변화 거부"
        
        analysis['later_fortune'] = {
            'title': "👵 말년운 (턱/하관)",
            'feature': f"턱선이 **매우** 발달하고 하관이 길어 튼튼하고 안정적인 인상입니다. (길이: {lower_face_length:.2f})",
            'fortune': f"말년운과 건강운에 **{strength}**을 가집니다. **인내심과 지구력**이 강해 평안하고 풍요로운 노후가 보장됩니다. 다만, **{risk}** 성향이 강해 자손 세대와의 소통에 어려움을 겪을 수 있습니다."
        }
    elif lower_face_length > 0.35: # 긴 경우
        strength = "안정적입니다"
        risk = "고집이 세지고 융통성 부족"
        
        analysis['later_fortune'] = {
            'title': "👵 말년운 (턱/하관)",
            'feature': f"턱선이 발달하고 하관이 길어 튼튼하고 안정적인 인상입니다. (길이: {lower_face_length:.2f})",
            'fortune': f"말년운이 **{strength}**. 중년의 성과를 잘 지켜내며 인복이 따릅니다. 다만, **{risk}**으로 인해 주변과의 의견 충돌을 겪기 쉬우니, 개방적인 자세가 필요합니다."
        }
    else: # 짧거나 좁은 경우
        if lower_face_length < 0.3:
            urgency = "매우 필수적"
            risk_detail = "인복과 안정감이 **극히** 약해질"
        else:
            urgency = "필수적"
            risk_detail = "안정감이 약해질"
            
        analysis['later_fortune'] = {
            'title': "👵 말년운 (턱/하관)",
            'feature': f"턱이 짧거나 좁은 편으로 활동적이고 민첩한 인상입니다. (길이: {lower_face_length:.2f})",
            'fortune': f"**활동적**이고 상황에 **민첩하게 대처하는 능력이 우수**합니다. 그러나 **인내심과 지구력이 부족**하여 말년의 복을 위해서는 **장기적인 재정 관리**가 **{urgency}**입니다. 턱 길이({lower_face_length:.2f})가 짧을수록 노후의 **{risk_detail}** 수 있으니, 젊을 때부터 인복을 쌓아야 합니다."
        }

    return analysis


# --- 3. GUI 및 카메라 통합 로직 (tkinter / OpenCV) ---

class FaceAnalysisApp:
    def __init__(self, window, window_title="Gemini 최종 전문가 관상 분석기"):
        self.window = window
        self.window.title(window_title)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('맑은 고딕', 10, 'bold'), padding=[10, 5])

        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
             messagebox.showerror("오류", "카메라를 찾을 수 없습니다. (카메라 연결 또는 권한 확인 필요)")
             self.window.destroy()
             return

        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.canvas = tk.Canvas(window, width=width, height=height, bg="black")
        self.canvas.pack(padx=10, pady=10)

        self.btn_capture=tk.Button(window, text="⭐ 관상 분석 시작 (삼정/오악/기색 통합 분석)", width=50, command=self.capture_and_analyze, 
                                   font=("맑은 고딕", 12, "bold"), fg="white", bg="#8B4513")
        self.btn_capture.pack(anchor=tk.CENTER, expand=True, pady=(0, 10))

        self.notebook = ttk.Notebook(window, width=width, height=280)
        self.notebook.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        self.tabs = {}
        tab_names = ["종합/전문 분석", "🥇 초년운 분석", "💵 중년운 분석", "👵 말년운 분석"]
        for name in tab_names:
            frame = ttk.Frame(self.notebook, padding="5 5 5 5")
            self.notebook.add(frame, text=name)
            
            text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("맑은 고딕", 10))
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.config(state=tk.DISABLED)
            
            self.tabs[name] = text_widget
        
        self.update_result_tab("종합/전문 분석", "얼굴을 정면에 맞추고 버튼을 눌러주세요.\n\n분석 중 얼굴에 특징점(노란색 선)이 실시간으로 표시됩니다.")

        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    # --- 보조 함수 ---
    def update_result_tab(self, tab_name, text):
        widget = self.tabs.get(tab_name)
        if widget:
            widget.config(state=tk.NORMAL)
            widget.delete('1.0', tk.END)
            widget.insert(tk.END, text)
            widget.config(state=tk.DISABLED)

    def format_section(self, analysis, key):
        item = analysis.get(key, {})
        text = f"=================================\n"
        text += f" {item.get('title', '분석 오류')} - 상세 해설\n"
        text += f"=================================\n"
        text += f"\n특징: {item.get('feature', '분석 실패')}\n"
        text += f"\n⭐ 전문가 상세 해설 (길흉 통합) ⭐\n{item.get('fortune', '분석 데이터가 불충분합니다.')}\n"
        return text
    # --- 보조 함수 끝 ---

    def update(self):
        ret, frame = self.vid.read()

        if ret:
            frame = cv2.flip(frame, 1) 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION, 
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) 
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

        self.window.after(self.delay, self.update)

    def capture_and_analyze(self):
        ret, frame = self.vid.read()
        if not ret:
            self.update_result_tab("종합/전문 분석", "⚠️ 카메라 프레임 캡처에 실패했습니다.")
            return
        
        frame = cv2.flip(frame, 1)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

        if not results.multi_face_landmarks:
            self.update_result_tab("종합/전문 분석", "⚠️ 얼굴이 감지되지 않았습니다. 정면을 바라보고 다시 시도해 주세요.")
            return
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        analysis = analyze_physiognomy(landmarks, frame)

        # 1. 종합/전문 분석 탭 내용 구성
        general_text = "=================================\n"
        general_text += "⭐ Gemini 최종 전문가 분석 (종합) ⭐\n"
        general_text += "=================================\n"
        
        for key in ['three_stops', 'prominence', 'tone']:
             item = analysis.get(key, {})
             general_text += f"\n[ {item.get('title', '정보 없음')} ]\n"
             general_text += f" - 특징: {item.get('feature', '정보 없음')}\n"
             general_text += f" - 해설: {item.get('fortune', '정보 없음')}\n"
             
        self.update_result_tab("종합/전문 분석", general_text)

        # 2. 초년/중년/말년 탭 내용 구성 (상세 해설)
        self.update_result_tab("🥇 초년운 분석", self.format_section(analysis, 'early_fortune'))
        self.update_result_tab("💵 중년운 분석", self.format_section(analysis, 'middle_fortune'))
        self.update_result_tab("👵 말년운 분석", self.format_section(analysis, 'later_fortune'))

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

# --- 4. 애플리케이션 실행 ---

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)

import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from mediapipe.python.solutions import hands, drawing_utils
from streamlit_drawable_canvas import st_canvas
from concurrent.futures import ThreadPoolExecutor
import re

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class MathWaveAI:
    ''' INITIALIZING THE CLASS '''
    def __init__(self):
        self.cap = None # Initialize camera to None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.imgCanvas = np.zeros((720, 1280, 3), dtype=np.uint8) # Blank Canvas

        self.p1, self.p2 = 0, 0 # Initial points for drawing (start and end points)
        self.fingers = [] # List to track which fingers are up : Will store 1 (up) or 0 (down) for each finger
        self.landmark_list = [] # Coordinates of hand landmarks
        # Detect 1 hand with atleast 75% confidence
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
        if "result" not in st.session_state:
            st.session_state.result = None
        if "processing" not in st.session_state:
            st.session_state.processing = False

    def streamlit_config(self):
        st.set_page_config(page_title="MathWave AI", layout="wide")
        st.markdown(
            """
            <style>
            [data-testid="stHeader"] { background: rgba(0,0,0,0); }
            .block-container { padding-top: 0rem; }
            .st-emotion-cache-1kyxreq { max-width: 100% !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<h1 style="text-align: center;">✍️ MathWave AI</h1>', unsafe_allow_html=True)

    def analyze_image(self, canvas_img): # Send image to gemini and get analysis
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = """Analyze the handwritten mathematical equation and provide the following in markdown format:
            
1. **Equation**: Write the recognized equation in LaTeX format (using $ delimiters)
2. **Solution Steps**: Show step-by-step solution process with explanations
3. **Final Answer**: Present the final answer clearly

Format your response like this:

**Equation**:  
$your_equation_here$

**Solution Steps**:  

**Final Answer**:  
$final_answer_here$"""
            response = model.generate_content([prompt, PIL.Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))]) # Convert OpenCV (BGR) to PIL (RGB)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def display_latex_output(self, result_text, placeholder):
        # Clear previous content
        placeholder.empty()
        
        # Process the response to maintain formatting
        processed_text = result_text.replace('**Equation**:', '### Equation')
        processed_text = processed_text.replace('**Solution Steps**:', '### Solution Steps')
        processed_text = processed_text.replace('**Final Answer**:', '### Final Answer')
        
        # Handle LaTeX content
        parts = re.split(r'(\$[^$]+\$)', processed_text)
        
        output = []
        for part in parts:
            if not part:
                continue
            if part.startswith('$') and part.endswith('$'):
                output.append(f'${part.strip("$")}$')
            else:
                output.append(part)
        
        # Display the formatted output
        placeholder.markdown('\n'.join(output), unsafe_allow_html=True)

    def run(self):
        self.streamlit_config()

        mode = st.radio("Select Drawing Mode:", ["Air Draw", "Trackpad Draw"], horizontal=True)

        if mode == "Trackpad Draw":
            self.trackpad_draw_ui()
        else:
            self.air_draw_ui()

    def trackpad_draw_ui(self): # Drawing interface using mouse/trackpad
        col1, _, col3 = st.columns([0.7, 0.05, 0.25])

        with col1:
            st.markdown("### Draw using trackpad/mouse")
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=7,
                stroke_color="black",
                background_color="#A9A9A9",
                height=720,
                width=1280,
                drawing_mode="freedraw",
                key="canvas_app",
            )

        with col3:
            st.markdown('<h5 style="color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty() # Placeholder for result

            if st.button("Analyze Drawing") and canvas_result.image_data is not None and not st.session_state.processing:
                st.session_state.processing = True
                gray = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2BGR) # Convert canvas to BGR
                result = self.analyze_image(gray) # Analyze using Gemini
                st.session_state.result = result # Store result
                st.session_state.processing = False  # Mark processing finished

            if st.session_state.result:
                self.display_latex_output(st.session_state.result, result_placeholder) # Show Result
                st.session_state.result = None

    def air_draw_ui(self):
        col1, _, col3 = st.columns([0.7, 0.05, 0.25])
        with col1:
            video_placeholder = st.empty()
        with col3:
            st.markdown('<h5 style="color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            success, img = self.cap.read() # Read frame from camera
            if not success:
                st.error("Camera Error - Please check webcam connection")
                break

            img = cv2.resize(img, (1280, 720))
            img = cv2.flip(img, 1) # Mirror Effect
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for mediapipe

            result = self.mphands.process(imgRGB) # Processes an RGB Image and Returns the Hand Landmarks
            self.landmark_list = [] # Reset Landmark List
            
            if result.multi_hand_landmarks: # If Hand Landmarks are detected
                for hand_landmarks in result.multi_hand_landmarks: 
                    drawing_utils.draw_landmarks(img, hand_landmarks, hands.HAND_CONNECTIONS)
                    for id, lm in enumerate(hand_landmarks.landmark): # Extract ID and Origin for Each Landmarks
                        h, w, _ = img.shape # Image height and width
                        self.landmark_list.append([id, int(lm.x * w), int(lm.y * h)]) # Store landmark id and xy coordinates (normalized coodinates to actual pixel positions)

            self.fingers = [] # Reset finger list to store which are up
            if self.landmark_list: # If landmark list is not empty
                for id in [4, 8, 12, 16, 20]:
                    if id != 4: # Check for each finger except the thumb is above lower joint (pt 2)
                        self.fingers.append(1 if self.landmark_list[id][2] < self.landmark_list[id - 2][2] else 0)
                    else: # thumb Finger : moves laterally
                        self.fingers.append(1 if self.landmark_list[id][1] < self.landmark_list[id - 2][1] else 0)

            if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1: 
                cx, cy = self.landmark_list[8][1], self.landmark_list[8][2] # Index finger tip
                if self.p1 or self.p2:
                    cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (255, 0, 255), 5)
                self.p1, self.p2 = cx, cy
            else:
                self.p1, self.p2 = 0, 0

            if self.fingers == [1, 0, 0, 0, 1]:
                self.imgCanvas = np.zeros_like(self.imgCanvas) 

            imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY) 
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) 
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR) 
            img = cv2.bitwise_and(img, imgInv) 
            img = cv2.bitwise_or(img, self.imgCanvas) 
            video_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

            if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1 and not st.session_state.processing: 
                st.session_state.processing = True 
                frozen_canvas = self.imgCanvas.copy() # Copy current canvas
                result = self.analyze_image(frozen_canvas) # Send to Gemini
                st.session_state.result = result
                st.session_state.processing = False
                self.imgCanvas = np.zeros_like(self.imgCanvas) # Clear canvas after processing

            if st.session_state.result:
                self.display_latex_output(st.session_state.result, result_placeholder) # Display output
                st.session_state.result = None
                self.imgCanvas = np.zeros_like(self.imgCanvas) # Clear again after showing

        self.cap.release() # Release camera


if __name__ == "__main__":
    try:
        app = MathWaveAI() # Create instance of the app
        app.run() # Run the app
    except Exception as e:
        st.error(f"Application Error: {str(e)}")

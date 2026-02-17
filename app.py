import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from processors import BioAnalyticEngine, ChromaticEngine
from logic import ComplianceController
import config

# ==========================================
# 1. Page & CSS Setup
# ==========================================
st.set_page_config(
    page_title=config.PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Injection
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fff; }
    
    /* Card Component */
    .info-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Metric Typography */
    .metric-head { font-size: 0.85rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
    .metric-body { font-size: 1.8rem; font-weight: 700; color: #60a5fa; margin-top: 5px; }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background-color: #065f46;
        color: #34d399;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Engine
chromatic_engine = ChromaticEngine()

# ==========================================
# 2. Helper Functions
# ==========================================
def render_radar_chart(emotions):
    """Creates a polar chart for emotion probability."""
    categories = list(emotions.keys())
    values = list(emotions.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line_color=config.THEME_COLOR, name='Emotion'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    return fig

# ==========================================
# 3. Main Application Flow
# ==========================================
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("🧬 " + config.PAGE_TITLE)
        st.markdown("Automated Biometric Classification System")
        st.divider()
        uploaded_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
        st.info("System Ready | Engine: VGG-Face")

    # --- Main Content ---
    if not uploaded_file:
        st.markdown(f"""
            <div style="text-align:center; padding: 50px;">
                <h1>Welcome to Bio-Identity</h1>
                <p style="color:#9ca3af;">Please upload a clear portrait image to begin analysis.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Process Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col_preview, col_results = st.columns([1, 1.5])

    # 1. Preview Column
    with col_preview:
        st.subheader("Subject Preview")
        st.image(img_rgb, use_container_width=True, caption="Input Data")

    # 2. Results Column
    with col_results:
        st.subheader("Analysis Report")
        
        with st.spinner("Initializing Deep Neural Networks..."):
            # Step A: AI Analysis
            biometrics = BioAnalyticEngine.scan_subject(img_rgb)
            
            if biometrics:
                # Step B: Apply Business Logic
                nationality = ComplianceController.derive_nationality(biometrics)
                rules = ComplianceController.get_visibility_rules(nationality)
                
                # Step C: Conditional Processing (Dress Color)
                dress_name, dress_hex = "RESTRICTED", "#333"
                if rules['show_dress']:
                    dress_name, dress_hex = chromatic_engine.analyze_attire(img_bgr, biometrics['region'])

                # --- RENDER DASHBOARD ---
                st.markdown(f"### Detected Profile: {nationality}")
                st.markdown(f"<span class='status-badge'>Confidence: High</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics Grid
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    emo_val = biometrics['dominant_emotion'].upper()
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="metric-head">Emotion</div>
                            <div class="metric-body">{emo_val}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    age_val = f"{biometrics['age']}" if rules['show_age'] else "N/A"
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="metric-head">Est. Age</div>
                            <div class="metric-body">{age_val}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with c3:
                    dress_val = dress_name if rules['show_dress'] else "N/A"
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="metric-head">Dress Color</div>
                            <div class="metric-body" style="color:{dress_hex}">{dress_val}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                
                # Emotion Chart
                if rules['show_emotion']:
                    st.markdown("#### Emotional Spectrum")
                    st.plotly_chart(render_radar_chart(biometrics['emotion']), use_container_width=True)
            
            else:
                st.error("⚠️ Analysis Failed: No face detected in the frame.")

if __name__ == "__main__":
    main()
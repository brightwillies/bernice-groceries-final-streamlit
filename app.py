import streamlit as st
import requests
from PIL import Image
import io
import base64

# Backend API URL (Update this with your Render URL after deployment)
BACKEND_URL = "https://your-yolo-backend.onrender.com"  # Change this!

st.title("YOLOv11 Grocery Item Detector")
st.write("Detects **cheerios**, **soup**, and **candle**.")

# Health check
try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=10)
    if health_response.status_code == 200:
        st.sidebar.success("✅ Backend connected")
    else:
        st.sidebar.error("❌ Backend not responding")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to backend: {e}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Objects"):
        st.write("Sending to backend for detection...")
        
        try:
            # Prepare file for upload
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            # Send to backend
            with st.spinner("Detecting objects..."):
                response = requests.post(
                    f"{BACKEND_URL}/predict", 
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display annotated image
                annotated_bytes = bytes.fromhex(result["annotated_image"])
                annotated_image = Image.open(io.BytesIO(annotated_bytes))
                st.image(annotated_image, caption="Detection Result", use_column_width=True)
                
                # Display detection details
                st.subheader("Detection Details")
                for detection in result["detections"]:
                    st.write(
                        f"- **Class:** {detection['class']}  |  "
                        f"**Confidence:** {detection['confidence']:.2f} | "
                        f"**Box:** {detection['bbox']}"
                    )
                    
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. The backend might be starting up.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Please check if the backend is running.")
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")

st.markdown("---")
st.caption("YOLOv11 Streamlit App – by Bernice Nhyira Eghan")
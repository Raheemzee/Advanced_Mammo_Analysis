import torch
import torch.nn as nn
import numpy as np
import pydicom
import cv2
from flask import Flask, render_template, request, send_file
import os, csv
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3

app = Flask(__name__)

UPLOAD = "uploads"
RESULT = "results"
REPORT = "reports"

for f in [UPLOAD, RESULT, REPORT]:
    os.makedirs(f, exist_ok=True)

# ================= AI MODEL =================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/model.pt"

CLASS_NAMES = [
    "BI-RADS A",
    "BI-RADS B",
    "BI-RADS C",
    "BI-RADS D"
]

def load_model():
    model = inception_v3(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 4)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        if k.startswith("net."):
            k = k.replace("net.", "")
        if k.startswith("model."):
            k = k.replace("model.", "")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def preprocess_for_model(image):
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)

def predict_density_ai(image):
    input_tensor = preprocess_for_model(image).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))

    return CLASS_NAMES[idx], float(probs[idx])

# ================= CORE =================

def analyze_mammogram(path, out, heat, mask_path):
    img = cv2.imread(path)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(3.0,(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced,(5,5),0)

    # Breast mask
    _, thresh = cv2.threshold(blur, 10,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 3)

    breast = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # ================= AI PREDICTION =================
    ai_label, ai_conf = predict_density_ai(breast)

    # ================= DENSITY (RULE BASED) =================
    dense = np.sum(breast > 180)
    total = np.sum(mask > 0)
    density_ratio = dense/(total+1e-5)

    if density_ratio < 0.25: density="Low"
    elif density_ratio < 0.5: density="Medium"
    else: density="High"

    # ================= SUSPICIOUS REGIONS =================
    _, suspicious = cv2.threshold(breast,200,255,cv2.THRESH_BINARY)
    suspicious = cv2.morphologyEx(suspicious, cv2.MORPH_OPEN, kernel,2)

    contours,_ = cv2.findContours(suspicious,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    region_data=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 8000:
            mask_temp = np.zeros(gray.shape,dtype="uint8")
            cv2.drawContours(mask_temp,[cnt],-1,255,-1)

            intensity = np.mean(breast[mask_temp==255])
            score = (area/1000) + (intensity/255)

            region_data.append({
                "area": int(area),
                "intensity": float(intensity),
                "score": round(score,2)
            })

            cv2.drawContours(original,[cnt],-1,(0,0,255),2)

    region_count = len(region_data)
    avg_size = np.mean([r["area"] for r in region_data]) if region_data else 0

    # ================= RISK =================
    risk_score = density_ratio*2 + region_count*0.3 + avg_size/5000

    if risk_score < 1: risk="Low"
    elif risk_score < 2: risk="Moderate"
    else: risk="High"

    # ================= VISUAL =================
    heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    cv2.imwrite(heat, heatmap)

    final_mask = np.zeros(gray.shape, dtype="uint8")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 8000:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    cv2.imwrite(mask_path, final_mask)

    # Overlay AI + traditional results
    cv2.putText(original,f"Risk:{risk}",(20,40),0,1,(0,0,255),2)
    cv2.putText(original,f"Density:{density}",(20,80),0,1,(255,255,0),2)
    cv2.putText(original,f"AI:{ai_label}",(20,120),0,1,(0,255,0),2)

    cv2.imwrite(out, original)

    return {
        "density": density,
        "density_ratio": round(density_ratio,3),
        "ai_density": ai_label,
        "ai_confidence": round(ai_conf,3),
        "regions": region_count,
        "avg_size": int(avg_size),
        "risk": risk,
        "region_data": region_data
    }

# ================= REPORT =================

def generate_report(results):
    path = os.path.join(REPORT,"report.csv")

    with open(path,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image","Density","AI Density","AI Confidence","Risk","Regions","Avg Size"
        ])

        for r in results:
            writer.writerow([
                r["name"],
                r["result"]["density"],
                r["result"]["ai_density"],
                r["result"]["ai_confidence"],
                r["result"]["risk"],
                r["result"]["regions"],
                r["result"]["avg_size"]
            ])

    return path

# ================= ROUTES =================

@app.route("/",methods=["GET","POST"])
def index():
    results=[]
    report=None

    if request.method=="POST":
        files=request.files.getlist("images")

        for file in files:
            name=secure_filename(file.filename)

            up=os.path.join(UPLOAD,name)
            out=os.path.join(RESULT,name)
            heat=os.path.join(RESULT,"heat_"+name)
            mask=os.path.join(RESULT,"mask_"+name)

            file.save(up)

            res=analyze_mammogram(up,out,heat,mask)

            results.append({
                "name":name,
                "heat":"heat_"+name,
                "mask":"mask_"+name,
                "result":res
            })

        report=generate_report(results)

    return render_template("index.html",results=results,report=report)

@app.route("/download")
def download():
    return send_file(os.path.join(REPORT,"report.csv"),as_attachment=True)

@app.route("/uploads/<f>")
def up(f): return send_file(os.path.join(UPLOAD,f))

@app.route("/results/<f>")
def res(f): return send_file(os.path.join(RESULT,f))

if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)

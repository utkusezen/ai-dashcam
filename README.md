# machine learning-driven traffic sign recognition and speed suggestion via mobile dashcam
This project incorporates traffic sign recognition and safe driving speeds suggestion into one project. It consists of a backend process and an android app.
Because of hardware limitations, this project does not solely run on a mobile phone, instead it requires a second device hosting the backend service. 

<img width="1024" height="768" alt="345" src="https://github.com/user-attachments/assets/a553601e-e50d-4300-b356-2ad936e7c674" />

Neural networks were trained to detect and classify traffic signs on any given image. Furthermore, visual computing methods are used to extract road and environment information, which are then inputted into another neural network. The predictions of the machine learning models are sent through an API to the android application and are displayed to the user.



<img width="1080" height="2248" alt="dashcam_app" src="https://github.com/user-attachments/assets/bbd31f05-9c6d-4d06-8563-49765b2277bb" />

 
## Installation Guide

### Prerequisites

- Python 3.10 (conda environment recommended)
- android smartphone
- devices running in the same wireless network

(Functionality tested with PyCharm)

---

### Run Backend

```bash
cd backend
pip install -r ..\requirements.txt
python app.py
```

---

### Android App

Install ai-dashcam.apk on an android phone.

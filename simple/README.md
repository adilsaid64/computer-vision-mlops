# Simple Deployment


Components: 
- Camera Input - Data Source - RTSP source
- RTSP Client - RTSP Clinet - Intake layer - Exposes a GET method that will return the latest camera frame 
- Inference Api - Model/Inference Layer - Poll the RTSP client for new frames to make inference on
- Event/Message Bus - Where events from Inference API get stored

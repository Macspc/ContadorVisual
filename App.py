import cv2
import mysql.connector
import numpy as np
import face_recognition
from datetime import datetime, timedelta

conn = mysql.connector.connect(
    host="162.241.2.193",
    user="macspc64_macspc",
    password="cacapava11c",
    database="macspc64_sensor"
)

count=0
# Verificar conexão
if conn.is_connected():
    print("Conexão bem-sucedida!")

cursor = conn.cursor()

# Criar tabela para armazenar as detecções de rosto
cursor.execute('''    CREATE TABLE IF NOT EXISTS faces_detected (
        id INT PRIMARY KEY AUTO_INCREMENT,
        timestamp DATETIME,
        image BLOB,
        person_count INT
    );''')
conn.commit()


# Função para salvar rostos detectados no banco de dados
def save_face_to_db(face_image, count):
    _, buffer = cv2.imencode('.jpg', face_image)
    image_bytes = buffer.tobytes()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    str1 = """
    INSERT INTO faces_detected (timestamp, image, person_count) 
     VALUES (%s, %s, %s)
     """
    cursor.execute(str1, (timestamp, image_bytes, person_count))
    conn.commit()

# Função para verificar se o rosto já foi identificado recentemente
def is_face_identified(face_encoding, known_encodings, threshold=0.6):
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=threshold)
    return True in matches

# Endereço da ESP32-CAM
url = 'http://192.168.0.245:81/stream'

# Iniciar captura de vídeo
webcam = cv2.VideoCapture(0)

# Executar algoritimo de detecção de face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Estrutura para armazenar encodings conhecidos com tempo de última detecção
known_face_encodings = []
face_last_seen_times = []

while True:
    ret, video = webcam.read()
    if not ret:
        print("Erro ao acessar a câmera. Verifique a conexão.")
        break

    # Flip para ajustar a orientação
    video = cv2.flip(video, 180)

    # Detectar rostos no vídeo
    faces = face_cascade.detectMultiScale(video, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))
    person_count = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 4)
        face_image = video[y:y+h, x:x+w]

        # Converter para RGB e codificar o rosto
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        if face_encodings:
            face_encoding = face_encodings[0]

            # Remover encodings com tempo limite excedido
            current_time = datetime.now()
            for i in range(len(face_last_seen_times) - 1, -1, -1):
                if current_time - face_last_seen_times[i] > timedelta(minutes=5):  # Tempo limite de 5 minutos
                    del known_face_encodings[i]
                    del face_last_seen_times[i]

            # Verificar se o rosto já foi identificado recentemente
            if not is_face_identified(face_encoding, known_face_encodings):
                save_face_to_db(face_image, person_count)
                known_face_encodings.append(face_encoding)
                face_last_seen_times.append(current_time)

    # Exibir o vídeo com os rostos detectados
    cv2.imshow("Face Detection", video)
    
    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
webcam.release()
cv2.destroyAllWindows()
conn.close()

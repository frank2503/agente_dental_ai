import os
import datetime
import pickle
import json
from key import get_key
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ---------- CONFIG ---------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SCOPES = ["https://www.googleapis.com/auth/calendar"]


# ---------- AUTENTICACIÓN GOOGLE CALENDAR ---------- #
def get_calendar_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials2.json", SCOPES)
            creds = flow.run_local_server(port=8080)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    service = build("calendar", "v3", credentials=creds)
    return service

# ---------- AGENTE CON LANGCHAIN (GROQ) ---------- #
def interpretar_cita(mensaje):
    llm = ChatGroq(model="llama-3.1-8b-instant")  # o llama-3-70b
    prompt = PromptTemplate(
        input_variables=["mensaje"],
        template="""
        Extrae la fecha y hora del siguiente mensaje de un paciente dental.
        Mensaje: "{mensaje}"

        Responde SOLO en formato JSON con las claves:
        - "fecha" en formato YYYY-MM-DD
        - "hora" en formato HH:MM (24 horas)
        """
    )
    response = llm.invoke(prompt.format(mensaje=mensaje))
    return response.content

# ---------- CREAR EVENTO ---------- #
def crear_evento(fecha, hora, descripcion="Cita dental"):
    service = get_calendar_service()

    start = datetime.datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M")
    end = start + datetime.timedelta(hours=1)

    event = {
        "summary": descripcion,
        "start": {"dateTime": start.isoformat(), "timeZone": "America/Mexico_City"},
        "end": {"dateTime": end.isoformat(), "timeZone": "America/Mexico_City"},
    }

    event = service.events().insert(calendarId="primary", body=event).execute()
    print(f"✅ Cita creada: {event.get('htmlLink')}")

# ---------- FLUJO PRINCIPAL ---------- #
if __name__ == "__main__":
    mensaje = input("Escribe el mensaje del paciente: ")
    datos = interpretar_cita(mensaje)

    print("📌 Datos interpretados:", datos)

    datos_limpios = datos.strip().replace("```json", "").replace("```", "").strip()

    try:
        datos_json = json.loads(datos_limpios)
        crear_evento(datos_json["fecha"], datos_json["hora"])
    except Exception as e:
        print("❌ Error interpretando la cita:", e)


import os
import datetime
import json
import dateparser
import re
import warnings
import pytz
from typing import List, Dict, Optional, Any, Union
from key import get_key
warnings.filterwarnings("ignore")

# --- Módulos de Google y LangChain ---
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# ---------- CONFIGURACIÓN ----------
GROQ_KEY = get_key()
if not GROQ_KEY:
    print("❌ ADVERTENCIA: No se encontró la API Key de Groq en tu archivo key.py.")
os.environ["GROQ_API_KEY"] = GROQ_KEY

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TIMEZONE = "America/Mexico_City"

HORARIO_APERTURA = datetime.time(9, 0)
HORARIO_CIERRE = datetime.time(18, 0)
DURACION_CITA = 60
INTERVALO_CITA = 30
DIAS_LABORALES = [0, 1, 2, 3, 4, 5]

# ---------- MEMORIA DE CONVERSACIÓN ----------
class DentalAssistantMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.reset_user_data()
        self.current_step = "saludo_inicial"

    def add_message(self, message: str, is_user: bool = True):
        self.memory.chat_memory.add_message(HumanMessage(content=message) if is_user else AIMessage(content=message))

    def get_chat_history(self) -> str:
        return self.memory.load_memory_variables({})['history']

    def update_user_data(self, field: str, value: Any):
        self.user_data[field] = value

    def get_user_data(self, field: str) -> Any:
        return self.user_data.get(field)
    
    def get_all_data(self) -> dict:
        return self.user_data

    def set_step(self, step: str):
        self.current_step = step

    def get_step(self) -> str:
        return self.current_step
    
    def reset_user_data(self):
        self.user_data = {
            "nombre": None, "email": None, "telefono": None,
            "fecha_cita": None, "hora_cita": None,
            "horarios_disponibles": None, "cita_agendada": False
        }

# ---------- AUTENTICACIÓN GOOGLE CALENDAR ----------
def get_calendar_service():
    creds = None
    token_path = "token.json"
    credentials_path = "credentials.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                if os.path.exists(token_path): os.remove(token_path)
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)

# ---------- FUNCIONES FECHA/HORA ----------
def parse_fecha_usuario(texto: str) -> Optional[datetime.datetime]:
    if not texto or texto.lower() == "preguntar_usuario": return None
    texto_limpio = texto.lower().strip().replace("el día ", "").replace("del ", "de ")
    fecha_dt = dateparser.parse(
        texto_limpio, languages=["es"],
        settings={"PREFER_DATES_FROM": "future", "TIMEZONE": TIMEZONE, "RETURN_AS_TIMEZONE_AWARE": True}
    )
    if not fecha_dt: return None
    if fecha_dt.year > datetime.datetime.now().year and str(datetime.datetime.now().year) not in texto:
        fecha_anio_actual = fecha_dt.replace(year=datetime.datetime.now().year)
        if fecha_anio_actual.date() >= datetime.datetime.now().date():
            return fecha_anio_actual
    return fecha_dt

# ---------- VALIDACIÓN DE DATOS ----------
def validar_email(email: str) -> bool:
    if not email: return False
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def validar_telefono(telefono: str) -> bool:
    if not telefono: return False
    return re.match(r'^(\+?\d{1,3}[- ]?)?\d{8,12}$', telefono.replace(" ", "").replace("-", "")) is not None

# ---------- EXTRAER INFORMACIÓN CON LLM ----------
def interpretar_mensaje(mensaje: str, memoria: DentalAssistantMemory) -> Dict[str, Any]:
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_KEY)
        prompt_text = f"""
Eres un asistente de una clínica dental. Tu tarea es extraer la INTENCIÓN y cualquier FECHA u HORA del mensaje del usuario.
Analiza el mensaje en el contexto del paso actual: "{memoria.get_step()}" y el historial.
Historial:
{memoria.get_chat_history()}

Responde SOLO con un objeto JSON válido con esta estructura:
{{
  "intencion": "agendar_cita | proporcionar_datos | corregir_informacion | confirmar | rechazar | despedirse | saludo | otro",
  "fecha": "texto con la fecha (ej: 'mañana', 'próximo lunes') o null",
  "hora": "texto con la hora (ej: '3 pm', '10:30') o null",
  "nombre": "nombre de la persona o null", "email": null, "telefono": null
}}

Ejemplos clave:
- Usuario: "Buen día, quisiera agendar una cita el día de mañana" -> {{"intencion": "agendar_cita", "fecha": "mañana", "hora": null, "nombre": null, "email": null, "telefono": null}}
- Usuario: "No, tiene que ser a las 3 pm" -> {{"intencion": "corregir_informacion", "fecha": null, "hora": "3 pm", "nombre": null, "email": null, "telefono": null}}
- Usuario: "Sí" -> {{"intencion": "confirmar", "fecha": null, "hora": null, "nombre": null, "email": null, "telefono": null}}

Mensaje del usuario: "{mensaje}"
"""
        response = llm.invoke(prompt_text)
        contenido = response.content if hasattr(response, 'content') else str(response)
        json_match = re.search(r'\{.*\}', contenido.strip().replace("```json", "").replace("```", "").strip(), re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"intencion": "error"}
    except Exception as e:
        print(f"Error interpretando mensaje: {e}")
        return {"intencion": "error"}

# ---------- LÓGICA DE CALENDARIO ----------
def obtener_horarios_disponibles(fecha: datetime.date) -> List[str]:
    try:
        service = get_calendar_service()
        tz = pytz.timezone(TIMEZONE)
        start_of_day = tz.localize(datetime.datetime.combine(fecha, HORARIO_APERTURA))
        end_of_day = tz.localize(datetime.datetime.combine(fecha, HORARIO_CIERRE))
        events_result = service.events().list(calendarId='primary', timeMin=start_of_day.isoformat(), timeMax=end_of_day.isoformat(), singleEvents=True, orderBy='startTime').execute()
        eventos = events_result.get('items', [])
        horarios_ocupados = set()
        for evento in eventos:
            inicio = datetime.datetime.fromisoformat(evento['start']['dateTime']).time()
            fin = datetime.datetime.fromisoformat(evento['end']['dateTime']).time()
            slot_actual = datetime.datetime.combine(fecha, inicio)
            while slot_actual.time() < fin:
                horarios_ocupados.add(slot_actual.strftime("%H:%M"))
                slot_actual += datetime.timedelta(minutes=INTERVALO_CITA)
        horarios_disponibles = []
        slot_actual = start_of_day
        while (slot_actual + datetime.timedelta(minutes=DURACION_CITA)).time() <= HORARIO_CIERRE:
            hora_str = slot_actual.strftime("%H:%M")
            if hora_str not in horarios_ocupados:
                horarios_disponibles.append(hora_str)
            slot_actual += datetime.timedelta(minutes=INTERVALO_CITA)
        return horarios_disponibles
    except Exception as e:
        print(f"Error al obtener horarios de Google Calendar: {e}")
        return []

def crear_evento_google(fecha_str: str, hora_str: str, nombre: str, email: str, telefono: str) -> str:
    try:
        service = get_calendar_service()
        tz = pytz.timezone(TIMEZONE)
        start_dt = tz.localize(datetime.datetime.strptime(f"{fecha_str} {hora_str}", "%Y-%m-%d %H:%M"))
        end_dt = start_dt + datetime.timedelta(minutes=DURACION_CITA)
        event = {"summary": f"Cita Dental - {nombre}", "description": f"Paciente: {nombre}\nEmail: {email}\nTeléfono: {telefono}", "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE}, "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE}, "attendees": [{"email": email}]}
        event = service.events().insert(calendarId="primary", body=event, sendUpdates="all").execute()
        return event.get("htmlLink", "Evento creado (sin link)")
    except Exception as e:
        print(f"Error creando evento: {e}")
        return "Error al agendar. Por favor, contacta a la clínica."

# ---------- FUNCIONES AUXILIARES ----------
# --- ESTA ES LA FUNCIÓN FINAL, CORREGIDA Y ROBUSTA ---
def procesar_seleccion_horario(respuesta: str, horarios_disponibles: List[str]) -> Optional[Union[str, list]]:
    respuesta_lower = respuesta.lower()
    TIME_KEYWORDS = ["las", "hora", "tarde", "mañana", "am", "pm", ":"]
    
    # Prioridad 1: Si el usuario usa lenguaje natural para la hora, procesarlo como tal.
    if any(keyword in respuesta_lower for keyword in TIME_KEYWORDS):
        parsed_time = dateparser.parse(respuesta, languages=['es'])
        if parsed_time:
            hora = parsed_time.hour
            # Inteligencia contextual: si la hora es < 9, es muy probable que sea PM.
            if hora < HORARIO_APERTURA.hour:
                hora += 12
            
            hora_str_buscada = str(hora).zfill(2)
            coincidencias = [h for h in horarios_disponibles if h.startswith(hora_str_buscada)]
            
            if len(coincidencias) == 1: return coincidencias[0]
            if len(coincidencias) > 1: return coincidencias
    
    # Prioridad 2 (Fallback): Si no parece una hora, intentar procesar como número de la lista.
    try:
        numero_limpio = re.sub(r'\D', '', respuesta)
        if numero_limpio:
            idx = int(numero_limpio)
            if 1 <= idx <= len(horarios_disponibles):
                return horarios_disponibles[idx - 1]
    except (ValueError, IndexError):
        pass

    return None

def formatear_fecha_amigable(fecha_str: str) -> str:
    fecha = datetime.datetime.strptime(fecha_str, "%Y-%m-%d")
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    dias_semana = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    return f"{dias_semana[fecha.weekday()]} {fecha.day} de {meses[fecha.month - 1]} de {fecha.year}"

def formatear_hora_amigable(hora_str: str) -> str:
    return datetime.datetime.strptime(hora_str, "%H:%M").strftime("%I:%M %p")

# ---------- FLUJO PRINCIPAL ----------
def main():
    print("👩‍⚕️ Hola, soy tu asistente dental virtual. Estoy aquí para ayudarte a agendar tu cita 🦷.")
    print("Puedes decir cosas como 'Quiero una cita para mañana' o 'Necesito ver horarios disponibles'.\n")
    
    memoria = DentalAssistantMemory()
    
    while True:
        try:
            mensaje = input("💬 Tú: ").strip()
            if not mensaje: continue
            
            memoria.add_message(mensaje)
            respuesta = ""
            step = memoria.get_step()
            interpretacion = interpretar_mensaje(mensaje, memoria)
            intencion = interpretacion.get("intencion")

            # --- Manejo del flujo principal ---
            if intencion == "despedirse":
                print("👋 ¡Gracias por contactarnos! Que tengas un excelente día.")
                break

            if step == "eligiendo_horario":
                resultado_seleccion = procesar_seleccion_horario(mensaje, memoria.get_user_data("horarios_disponibles"))
                if isinstance(resultado_seleccion, str):
                    memoria.update_user_data("hora_cita", resultado_seleccion)
                    fecha = memoria.get_user_data("fecha_cita")
                    respuesta = f"✅ Perfecto. ¿Confirmas tu cita para el {formatear_fecha_amigable(fecha)} a las {formatear_hora_amigable(resultado_seleccion)}? (Sí/No)"
                    memoria.set_step("confirmar_cita")
                elif isinstance(resultado_seleccion, list):
                    respuesta = "Tengo varias opciones con esa hora. ¿A cuál te refieres?\n"
                    for i, horario_ambiguo in enumerate(resultado_seleccion, 1):
                        respuesta += f"    {i}. {formatear_hora_amigable(horario_ambiguo)}\n"
                    memoria.update_user_data("horarios_disponibles", resultado_seleccion)
                else:
                    respuesta = "❌ No reconocí ese horario. Por favor, elige uno de la lista."
            
            elif step == "confirmar_cita":
                if intencion == 'confirmar':
                    respuesta = "👍 ¡Genial! Para terminar, ¿podrías darme tu nombre completo, email y teléfono?"
                    memoria.set_step("solicitar_datos_personales")
                # --- LÓGICA DE CORRECCIÓN MEJORADA ---
                elif intencion == 'corregir_informacion' and interpretacion.get('hora'):
                    respuesta = f"Entendido, cambiemos la hora. Verificando disponibilidad para las {interpretacion.get('hora')}..."
                    # Se simula un re-procesamiento con la nueva hora.
                    resultado_seleccion = procesar_seleccion_horario(interpretacion.get('hora'), memoria.get_user_data("horarios_disponibles"))
                    if isinstance(resultado_seleccion, str):
                        memoria.update_user_data("hora_cita", resultado_seleccion)
                        fecha = memoria.get_user_data("fecha_cita")
                        respuesta += f"\n✅ Perfecto. ¿Confirmas tu cita para el {formatear_fecha_amigable(fecha)} a las {formatear_hora_amigable(resultado_seleccion)}? (Sí/No)"
                        memoria.set_step("confirmar_cita") # Permanece en el mismo paso de confirmación
                    else:
                        respuesta = "❌ Lo siento, esa hora no está disponible. ¿Te gustaría elegir otra de la lista?"
                        memoria.set_step("eligiendo_horario")
                else:
                    respuesta = "De acuerdo. ¿Para qué otra fecha te gustaría buscar?"
                    memoria.set_step("saludo_inicial")
            
            elif step == "confirmar_datos_personales":
                if intencion == 'confirmar':
                    datos = memoria.get_all_data()
                    link = crear_evento_google(datos['fecha_cita'], datos['hora_cita'], datos['nombre'], datos['email'], datos['telefono'])
                    respuesta = f"✅ ¡Perfecto! Tu cita ha sido agendada. Se ha enviado una invitación a tu correo.\n📅 Detalles: {link}\n\n¿Puedo ayudarte en algo más?"
                    memoria.update_user_data("cita_agendada", True)
                    memoria.set_step("finalizado")
                else:
                    respuesta = "¿Qué dato te gustaría corregir? (nombre, email o teléfono)"
                    memoria.set_step("solicitar_datos_personales")

            elif intencion == "proporcionar_datos" or step == "solicitar_datos_personales":
                for key in ["nombre", "email", "telefono"]:
                    if interpretacion.get(key) and not memoria.get_user_data(key):
                         memoria.update_user_data(key, interpretacion[key])
                datos = memoria.get_all_data()
                if not datos['nombre']:
                    respuesta = "Entendido. ¿Cuál es tu nombre completo?"
                elif not datos['email'] or not validar_email(datos['email']):
                    respuesta = f"Gracias, {datos['nombre'].split()[0]}. ¿Cuál es tu email para la confirmación?"
                elif not datos['telefono'] or not validar_telefono(datos['telefono']):
                    respuesta = "Casi terminamos. ¿Cuál es tu teléfono de contacto?"
                else:
                    respuesta = f"Gracias. Revisa si tus datos son correctos:\n- Nombre: {datos['nombre']}\n- Email: {datos['email']}\n- Teléfono: {datos['telefono']}\n\n¿Es correcto? (Sí/No)"
                    memoria.set_step("confirmar_datos_personales")
            
            elif intencion in ["agendar_cita", "corregir_informacion", "saludo"]:
                fecha_texto = interpretacion.get("fecha")
                if not fecha_texto:
                    respuesta = "Claro, ¿para qué fecha te gustaría tu cita?"
                else:
                    fecha_dt = parse_fecha_usuario(fecha_texto)
                    if not fecha_dt:
                        respuesta = f"❌ No entendí la fecha '{fecha_texto}'. Prueba con 'mañana' o '15 de septiembre'."
                    elif fecha_dt.weekday() not in DIAS_LABORALES:
                        respuesta = f"Lo siento, solo abrimos de Lunes a Sábado. El {formatear_fecha_amigable(fecha_dt.strftime('%Y-%m-%d'))} está cerrado."
                    else:
                        horarios = obtener_horarios_disponibles(fecha_dt.date())
                        if not horarios:
                            respuesta = f"Lo siento, no tengo horarios para el {formatear_fecha_amigable(fecha_dt.strftime('%Y-%m-%d'))}. ¿Quieres buscar otra fecha?"
                        else:
                            fecha_str = fecha_dt.strftime("%Y-%m-%d")
                            memoria.update_user_data("fecha_cita", fecha_str)
                            memoria.update_user_data("horarios_disponibles", horarios)
                            respuesta = f"⏰ Horarios disponibles para el {formatear_fecha_amigable(fecha_str)}:\n"
                            for i, h in enumerate(horarios, 1):
                                respuesta += f"    {i}. {formatear_hora_amigable(h)}\n"
                            respuesta += "\n¿Qué horario prefieres?"
                            memoria.set_step("eligiendo_horario")
            
            elif step == "finalizado":
                memoria.reset_user_data()
                memoria.set_step("saludo_inicial")
                respuesta = "¡Claro! ¿Para qué fecha sería tu nueva cita?"
            
            elif intencion != "error":
                respuesta = "Disculpa, no te entendí. Puedo ayudarte a agendar una cita."

            if respuesta:
                print(f"👩‍⚕️ {respuesta}")
                memoria.add_message(respuesta, is_user=False)
        except Exception as e:
            print(f"❌ Lo siento, ocurrió un error inesperado: {e}. Vamos a intentarlo de nuevo.")
            memoria = DentalAssistantMemory()

if __name__ == "__main__":
    main()
# Rostro a rostro

Web de intercambio facial con detección automática de rostro mediante InsightFace `buffalo_l` (también guardado como `buffalo_1` por algunas aplicaciones). La detección y los 106 puntos faciales se calculan en el backend; la deformación geométrica y la composición se ejecutan en el navegador.

Las imágenes se transmiten temporalmente al backend de Render para detectar puntos. El código no escribe las imágenes a disco ni conserva uploads. Usa fotos propias o con permiso. Los modelos preentrenados de InsightFace son para investigación no comercial; revisa con el proveedor la licencia aplicable antes de un uso comercial o público.

## Ejecutar localmente

Con Python 3.11 y dependencias instaladas (`pip install -r requirements.txt`), inicia el servidor:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000`. En el primer inicio, InsightFace descarga el paquete de modelos `buffalo_l`.

## Publicar en Render

Sube el contenido de esta carpeta a la raíz de un repositorio GitHub y crea un Blueprint en Render seleccionando el archivo `render.yaml`. El servicio requiere Python 3.11. El modelo se descarga al iniciar el servicio. Render debe tener permiso de salida para descargar el paquete de InsightFace.

El servicio escucha en `0.0.0.0:$PORT` y expone `/healthz` para los controles de salud.

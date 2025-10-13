# Proyecto-1---BI

## Configuración de clave de OpenAI (Windows)

Este repositorio requiere una clave de OpenAI para la aumentación de datos en `etapa2.ipynb`. Para mantener la seguridad, ya no se guarda la clave en el código: se toma de la variable de entorno `OPENAI_API_KEY`.

### Pasos rápidos

1. Crea un archivo `.env` en la raíz del proyecto con el contenido:

```
OPENAI_API_KEY=tu_clave_aqui
```

2. Carga esa variable en tu sesión de PowerShell antes de ejecutar Jupyter:

```powershell
$env:OPENAI_API_KEY = (Get-Content .env | Select-String -Pattern '^OPENAI_API_KEY=' | ForEach-Object { $_.ToString().Split('=')[1] })
jupyter notebook
```

O, si prefieres dejarla persistente en tu perfil de PowerShell:

```powershell
[Environment]::SetEnvironmentVariable('OPENAI_API_KEY','tu_clave_aqui','User')
```

3. Ejecuta `etapa2.ipynb`. La celda que usa OpenAI ahora lee la clave desde `OPENAI_API_KEY` y fallará con un mensaje claro si no está definida.

### Notas de seguridad

- Nunca subas tu clave al repositorio. `.env` ya está ignorado en `.gitignore`.
- Si una clave se expone, rótala inmediatamente en tu cuenta de OpenAI.

## Ejecutar la aplicación con Docker (API + UI)

La app (backend FastAPI y la interfaz web) corre en un único contenedor.

### Requisitos
- Docker Desktop (Windows, con WSL2 habilitado) y el Engine en estado "Running".

### Variables de entorno
Puedes usar un archivo `.env` en la raíz del proyecto (recomendado):

```
ADMIN_PASSWORD=tu-clave-admin
# Opcional (solo notebooks de aumento):
# OPENAI_API_KEY=tu_clave
```

### Levantar el servicio

```powershell
docker compose up --build
# Si tu versión usa el binario antiguo:
# docker-compose up --build
```

### Acceso
- UI (Home): `http://localhost:8000/`
- Modo admin: `http://localhost:8000/admin/login` (usa `ADMIN_PASSWORD`)
- OpenAPI/Docs: `http://localhost:8000/docs`

### Persistencia
`docker-compose.yml` monta como volúmenes locales las carpetas:
- `./artifacts` → modelos y métricas (p. ej., `artifacts/model_v2.pkl`)
- `./data` → datasets y `training_store.csv`

Esto permite que el modelo y los datos se conserven entre ejecuciones.

### Notas
- La interfaz web y el backend comparten el mismo proceso FastAPI; no necesitas servicios separados.
- Formatos soportados en la UI:
  - Predicción rápida: texto en formulario.
  - Predicción batch: archivo `.csv` o `.xlsx` con columna `textos`.
  - Reentrenar (admin): `.csv` o `.xlsx` con columnas `textos` y `labels` (valores 1, 3 o 4).

### Solución de problemas (Docker/WSL)
- Si recibes errores tipo overlayfs/`metadata.db` al crear el contenedor:
  1. Reinicia WSL y el servicio de Docker (PowerShell como Administrador):
     ```powershell
     wsl --shutdown
     net stop com.docker.service
     net start com.docker.service
     ```
  2. Limpia caché de Docker:
     ```powershell
     docker builder prune -af
     docker system prune -af --volumes
     ```
  3. Asegura espacio libre en disco (>10 GB) y WSL2 habilitado.
  4. Si persiste, prueba mover el proyecto fuera de OneDrive (p. ej., `C:\dev\Proyecto-1---BI`) y vuelve a ejecutar.

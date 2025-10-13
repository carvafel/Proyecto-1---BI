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
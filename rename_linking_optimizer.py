"""
Script temporal para renombrar linking-optimizer a linking_optimizer.
Intenta varias estrategias si el directorio está bloqueado.
"""

import os
import sys
import shutil
import time
from pathlib import Path

def rename_with_retry(old_path, new_path, max_attempts=5):
    """Intenta renombrar con reintentos."""
    for attempt in range(max_attempts):
        try:
            if new_path.exists():
                print(f"✅ {new_path.name} ya existe (ya fue renombrado)")
                return True

            print(f"Intento {attempt + 1}/{max_attempts}: Renombrando {old_path.name} -> {new_path.name}")
            old_path.rename(new_path)
            print(f"✅ Renombrado exitosamente!")
            return True

        except PermissionError as e:
            print(f"❌ Error: {e}")
            if attempt < max_attempts - 1:
                print(f"   Esperando 2 segundos antes del siguiente intento...")
                time.sleep(2)
            else:
                print(f"\n⚠️  No se pudo renombrar después de {max_attempts} intentos")
                print(f"   Posibles causas:")
                print(f"   - VSCode está abierto con este directorio")
                print(f"   - Hay un proceso de Python/Streamlit corriendo")
                print(f"   - El Explorador de Windows está en ese directorio")
                print(f"\n   Solución:")
                print(f"   1. Cierra VSCode")
                print(f"   2. Cierra cualquier ventana del Explorador en esa carpeta")
                print(f"   3. Ejecuta este script de nuevo")
                return False

        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False

    return False

def main():
    print("="*70)
    print("RENOMBRAR DIRECTORIO: linking-optimizer -> linking_optimizer")
    print("="*70)
    print()

    apps_dir = Path(__file__).parent / "apps"
    old_path = apps_dir / "linking-optimizer"
    new_path = apps_dir / "linking_optimizer"

    if not old_path.exists() and new_path.exists():
        print("✅ El directorio ya está renombrado correctamente!")
        print(f"   {new_path} existe")
        return 0

    if not old_path.exists():
        print(f"❌ Error: No se encuentra el directorio {old_path}")
        return 1

    print(f"Directorio origen: {old_path}")
    print(f"Directorio destino: {new_path}")
    print()

    success = rename_with_retry(old_path, new_path)

    print()
    print("="*70)
    if success:
        print("✅ OPERACIÓN COMPLETADA EXITOSAMENTE")
        print()
        print("Próximos pasos:")
        print("1. Los módulos ahora pueden importar correctamente")
        print("2. Ejecuta: git add apps/linking_optimizer")
        print("3. Ejecuta: git commit -m 'refactor: renombrar linking-optimizer'")
    else:
        print("❌ OPERACIÓN FALLIDA")
        print()
        print("Por favor, cierra todos los programas que puedan estar usando")
        print("el directorio 'linking-optimizer' y ejecuta este script de nuevo.")
    print("="*70)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

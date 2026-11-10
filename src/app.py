"""Punto de entrada de la interfaz de usuario industrial (Streamlit).

Sigue el estándar de telemetría de misión crítica y reglas anti-AI slop
definidas en .agents/rules/industrial-ui-standard.md.
"""

from typing import Any

import streamlit as st

from src.config import BASE_DIR


def render_header() -> None:
    """Renderiza la cabecera técnica de la consola SCADA / Telemetría."""
    st.markdown(
        """
        <div style="background-color: #11131A; border: 1px solid #1E2230; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="color: #F3F4F6; margin: 0; font-family: 'Inter', sans-serif; font-size: 1.25rem;">
                        SMART ALPR IOT &bull; CONSOLA DE TELEMETRÍA Y ACCESO VIAL
                    </h2>
                    <p style="color: #9CA3AF; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                        Uniguajira / Viva Wajiira &bull; Riohacha, Colombia
                    </p>
                </div>
                <div>
                    <span style="background: rgba(16, 185, 129, 0.12); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                        SISTEMA ONLINE
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> dict[str, Any]:
    """Punto de ejecución de la aplicación web."""
    st.set_page_config(
        page_title="Smart ALPR IoT | Telemetría Vial",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_header()
    return {"status": "ok", "base_dir": str(BASE_DIR)}


if __name__ == "__main__":
    main()

🤖 Sistema Automatizado de Trading ETHUSD
Sistema completo de trading automatizado que combina predicciones LSTM con análisis técnico y ejecución en Kraken.
📋 Estructura del Sistema
├── ethusd_lstm.py              # Entrenamiento del modelo LSTM
├── predict_and_filter.py       # Predicciones + Filtros técnicos
├── kraken_trader.py            # Ejecución de órdenes en Kraken
├── trading_orchestrator.py     # Orquestador principal
├── analytics.py                # Análisis y reportes
├── schedule.yml                # GitHub Actions (deprecated)
├── trading_bot.yml             # Workflow actualizado
└── ETHUSD_MODELS/              # Modelos entrenados
🎯 Características
1. Predicciones LSTM

Modelo Multi-Output LSTM entrenado con datos de 1h
Predice: High, Low, Close de la próxima vela
Actualización del modelo diaria

2. Análisis Técnico

Descarga datos cada 5 minutos
Indicadores: MA, EMA, RSI, MACD, ATR, Bollinger Bands
Sistema de scoring multi-filtro para señales

3. Gestión de Órdenes

Ejecución automática en Kraken
Take Profit: 80% de la predicción
Stop Loss: 2x ATR
Cierre automático por TP/SL/Timeout (60 min)

4. Monitoreo

Revisión cada 15 minutos
Notificaciones Telegram en tiempo real
Registro completo en CSV

📊 Lógica de Trading
Señal BUY (Score ≥ 6/12)

✅ Predicción alcista > 0.5%
✅ Tendencia MA alcista (3 pts)
✅ Precio sobre MA20 (2 pts)
✅ MACD alcista (2 pts)
✅ RSI oversold o BB inferior (2 pts)
✅ Alto volumen (1 pt)
✅ Pred High > 1.5% (2 pts)

Señal SELL (Score ≥ 6/12)

❌ Predicción bajista < -0.5%
❌ Tendencia MA bajista (3 pts)
❌ Precio bajo MA20 (2 pts)
❌ MACD bajista (2 pts)
❌ RSI overbought o BB superior (2 pts)
❌ Alto volumen (1 pt)
❌ Pred Low < -1.5% (2 pts)

Gestión de Riesgo

TP: 80% del movimiento predicho
SL: 2x ATR desde entrada
Timeout: Cierre forzado a los 60 min
Confianza mínima: 60% para ejecutar

🔧 Configuración
1. Variables de Entorno (Kraken API)
pythonKRAKEN_API_KEY = "tu_api_key"
KRAKEN_API_SECRET = "tu_api_secret"
2. Telegram Bot
pythonTELEGRAM_API = 'tu_bot_token'
CHAT_ID = 'tu_chat_id'
3. Parámetros de Trading
pythonVOLUME = 0.01  # ETH por orden
TP_PERCENTAGE = 0.80  # 80% del target
ATR_MULTIPLIER = 2  # Para SL
MIN_CONFIDENCE = 60  # Mínimo para ejecutar
🚀 Uso
Modo Local (24/7)
bash# Instalar dependencias
pip install -r requirements.txt

# Ejecutar orquestador
python trading_orchestrator.py
Modo GitHub Actions
bash# Push al repositorio
git push origin main

# El workflow ejecutará:
# - Predicción: Cada hora
# - Trading: Tras cada predicción
# - Monitoreo: Cada 15 min
Análisis Manual
bash# Generar reporte completo
python analytics.py
📈 Archivos Generados
CSVs de Trading

trading_signals.csv: Todas las señales generadas
orders_executed.csv: Órdenes colocadas
kraken_trades.csv: Trades completados
open_orders.json: Órdenes activas

Estructura de trading_signals.csv
csvtimestamp, current_price, pred_high, pred_low, pred_close, 
pred_change_%, signal, confidence, reason, ma_20, rsi, macd, atr
Estructura de kraken_trades.csv
csvtimestamp, txid, side, entry_price, close_price, volume, 
tp, sl, close_reason, time_open_min, pnl_usd, pnl_%
📊 Dashboards
Métricas Principales

Win Rate: % de trades ganadores
Profit Factor: Ganancias / Pérdidas
P&L Total: Beneficio acumulado
Avg Time: Tiempo promedio por trade

Distribuciones

Curva de equity
P&L por día
Cierres por razón (TP/SL/Timeout)
Win/Loss por tipo (BUY/SELL)

🔔 Notificaciones Telegram
En cada predicción:
🔮 ETHUSD - Análisis Actualizado
💰 Precio: $3,245.67
🔮 Predicción: HIGH/LOW/CLOSE
🚦 Señal: BUY/SELL/HOLD
📊 Confianza: 75%
Al ejecutar orden:
🚀 Nueva Orden Ejecutada
📊 Tipo: BUY
💰 Entrada: $3,245.67
🎯 TP: $3,290.00
🛑 SL: $3,200.00
Al cerrar orden:
✅ Orden Cerrada
💵 P&L: $45.23 (+1.39%)
🎯 Razón: TP
Reporte diario (23:00):
📊 Reporte Diario
🔢 Trades: 5
✅ Win Rate: 80%
💰 P&L: $127.45
⚠️ Consideraciones Importantes
Riesgos

Volatilidad: Crypto es extremadamente volátil
Slippage: Diferencia entre precio esperado y ejecutado
Comisiones: Kraken cobra fees por trade
Model Drift: El modelo puede perder precisión con el tiempo

Recomendaciones

Empezar pequeño: Usa volúmenes bajos al inicio
Monitorear: Revisa métricas diariamente
Backtesting: Prueba con datos históricos antes de live
Diversificar: No pongas todo tu capital en un bot
Ajustar parámetros: Optimiza según tus resultados

Optimizaciones Futuras

 Backtesting automatizado
 Ajuste dinámico de TP/SL
 Múltiples timeframes
 Ensemble de modelos
 Trailing stop loss
 Gestión de posición variable
 Paper trading mode
 Web dashboard
 Alertas de anomalías

📚 Dependencias
txtpandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
scikit-learn>=1.3.0
yfinance>=0.2.28
matplotlib>=3.7.0
requests>=2.31.0
schedule>=1.2.0
joblib>=1.3.0
tqdm>=4.66.0
🛠️ Troubleshooting
Error: "No se puede conectar a Kraken"

Verifica API keys
Revisa permisos de la API
Comprueba límites de rate

Error: "Modelo no encontrado"

Ejecuta python ethusd_lstm.py primero
Verifica carpeta ETHUSD_MODELS/

Error: "Insufficient balance"

Revisa saldo en Kraken
Reduce VOLUME en el código

Señales no se ejecutan

Verifica MIN_CONFIDENCE
Revisa si hay órdenes abiertas
Comprueba logs de Telegram

📄 Licencia
MIT License - Usa bajo tu propio riesgo
⚖️ Disclaimer
IMPORTANTE: Este sistema es solo para propósitos educativos. El trading automatizado conlleva riesgos significativos. Nunca inviertas dinero que no puedas permitirte perder. Los rendimientos pasados no garantizan resultados futuros. El autor no se hace responsable de pérdidas financieras.

Desarrollado con 🤖 para trading automatizado de ETHUSD

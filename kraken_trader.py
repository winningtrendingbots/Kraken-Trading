import requests
import json
import hmac
import hashlib
import base64
import time
import urllib.parse
import pandas as pd
import os
from datetime import datetime

# Configuración Kraken
KRAKEN_API_KEY = "BuVj1zFpmH8aoKXBMCfvcfmso4FD7O5tAlXDFD9aLNDc91S1wXYqNXVs"
KRAKEN_API_SECRET = "XLDq0M9GmSgzjerQNiXhoq7QsHRPF2qaVowSq8He7kVrlyXnF1Lf59v3lGccCitkuki68FsJvv79idoT10OeEQ=="
KRAKEN_API_URL = "https://api.kraken.com"

# Telegram
TELEGRAM_API = '8286372753:AAF356kUIEbZRI-Crdo4jIrXc89drKGWIWY'
CHAT_ID = '5825443798'

# Archivos
TRADES_FILE = 'kraken_trades.csv'
OPEN_ORDERS_FILE = 'open_orders.json'

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

# Autenticación Kraken
def kraken_signature(urlpath, data, secret):
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

def kraken_request(uri_path, data):
    headers = {
        'API-Key': KRAKEN_API_KEY,
        'API-Sign': kraken_signature(uri_path, data, KRAKEN_API_SECRET)
    }
    req = requests.post(KRAKEN_API_URL + uri_path, headers=headers, data=data)
    return req.json()

# Obtener precio actual
def get_current_price():
    url = f"{KRAKEN_API_URL}/0/public/Ticker?pair=ETHUSD"
    r = requests.get(url).json()
    if 'result' in r and 'XETHZUSD' in r['result']:
        return float(r['result']['XETHZUSD']['c'][0])
    return None

# Obtener balance
def get_balance():
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/Balance', data)
    return result

# Abrir orden
def place_order(side, volume, price, tp_price, sl_price):
    """
    side: 'buy' o 'sell'
    volume: cantidad en ETH
    price: precio límite (None para market order)
    tp_price: take profit
    sl_price: stop loss
    """
    data = {
        'nonce': str(int(1000*time.time())),
        'ordertype': 'limit' if price else 'market',
        'type': side,
        'volume': str(volume),
        'pair': 'XETHZUSD',
    }
    
    if price:
        data['price'] = str(price)
    
    # Añadir TP y SL
    if tp_price and sl_price:
        data['close'] = {
            'ordertype': 'limit',
            'price': str(tp_price)
        }
        # Kraken no soporta SL directo, usar conditional close
        # Implementaremos monitoreo manual
    
    result = kraken_request('/0/private/AddOrder', data)
    return result

# Cerrar orden
def cancel_order(txid):
    data = {
        'nonce': str(int(1000*time.time())),
        'txid': txid
    }
    result = kraken_request('/0/private/CancelOrder', data)
    return result

# Obtener órdenes abiertas
def get_open_orders():
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/OpenOrders', data)
    return result

# Calcular TP y SL
def calculate_tp_sl(entry_price, side, atr, pred_high, pred_low, tp_percentage=0.80):
    """
    Calcula TP al 80% de la predicción y SL con ATR
    """
    if side == 'buy':
        # TP: 80% del movimiento hacia pred_high
        target_move = pred_high - entry_price
        tp = entry_price + (target_move * tp_percentage)
        # SL: 2x ATR por debajo
        sl = entry_price - (atr * 2)
    else:  # sell
        # TP: 80% del movimiento hacia pred_low
        target_move = entry_price - pred_low
        tp = entry_price - (target_move * tp_percentage)
        # SL: 2x ATR por encima
        sl = entry_price + (atr * 2)
    
    return round(tp, 2), round(sl, 2)

# Monitorear y gestionar órdenes
def monitor_orders():
    """Monitorea órdenes abiertas y cierra por TP/SL/tiempo"""
    if not os.path.exists(OPEN_ORDERS_FILE):
        return
    
    with open(OPEN_ORDERS_FILE, 'r') as f:
        orders = json.load(f)
    
    current_price = get_current_price()
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    updated_orders = []
    
    for order in orders:
        txid = order['txid']
        entry_price = order['entry_price']
        side = order['side']
        tp = order['tp']
        sl = order['sl']
        open_time = datetime.fromisoformat(order['open_time'])
        volume = order['volume']
        
        # Tiempo abierto (en minutos)
        time_open = (datetime.now() - open_time).total_seconds() / 60
        
        should_close = False
        close_reason = None
        close_price = current_price
        
        # Verificar TP
        if side == 'buy' and current_price >= tp:
            should_close = True
            close_reason = 'TP'
        elif side == 'sell' and current_price <= tp:
            should_close = True
            close_reason = 'TP'
        
        # Verificar SL
        elif side == 'buy' and current_price <= sl:
            should_close = True
            close_reason = 'SL'
        elif side == 'sell' and current_price >= sl:
            should_close = True
            close_reason = 'SL'
        
        # Verificar tiempo (cerrar después de 60 minutos)
        elif time_open >= 60:
            should_close = True
            close_reason = 'TIMEOUT'
        
        if should_close:
            print(f"🔴 Cerrando orden {txid} por {close_reason}")
            
            # Cancelar en Kraken
            cancel_result = cancel_order(txid)
            print(f"   Kraken cancel: {cancel_result}")
            
            # Calcular P&L
            if side == 'buy':
                pnl = (close_price - entry_price) * volume
                pnl_pct = ((close_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - close_price) * volume
                pnl_pct = ((entry_price - close_price) / entry_price) * 100
            
            # Guardar en CSV
            trade_data = {
                'timestamp': datetime.now(),
                'txid': txid,
                'side': side,
                'entry_price': entry_price,
                'close_price': close_price,
                'volume': volume,
                'tp': tp,
                'sl': sl,
                'close_reason': close_reason,
                'time_open_min': time_open,
                'pnl_usd': pnl,
                'pnl_%': pnl_pct
            }
            
            df = pd.DataFrame([trade_data])
            if os.path.exists(TRADES_FILE):
                df.to_csv(TRADES_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(TRADES_FILE, index=False)
            
            # Telegram
            emoji = "✅" if pnl > 0 else "❌"
            msg = f"""
{emoji} *Orden Cerrada*

📝 ID: {txid[:8]}...
📊 Tipo: {side.upper()}
💰 Entrada: ${entry_price:.2f}
💰 Salida: ${close_price:.2f}
🎯 Razón: {close_reason}
⏱️ Tiempo: {time_open:.1f} min

💵 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)
"""
            send_telegram(msg)
        else:
            updated_orders.append(order)
    
    # Actualizar archivo
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(updated_orders, f, indent=2)

# Ejecutar señal de trading
def execute_signal():
    """Lee última señal y ejecuta si es BUY/SELL"""
    
    # Leer última señal
    signals_file = 'trading_signals.csv'
    if not os.path.exists(signals_file):
        print("❌ No hay señales disponibles")
        return
    
    df = pd.read_csv(signals_file)
    latest = df.iloc[-1]
    
    signal = latest['signal']
    
    if signal == 'HOLD':
        print("⏸️ Señal HOLD - No hay acción")
        return
    
    # Verificar si ya hay orden abierta
    if os.path.exists(OPEN_ORDERS_FILE):
        with open(OPEN_ORDERS_FILE, 'r') as f:
            open_orders = json.load(f)
        if len(open_orders) > 0:
            print("⚠️ Ya hay una orden abierta")
            return
    
    # Obtener datos necesarios
    current_price = get_current_price()
    atr = latest['atr']
    pred_high = latest['pred_high']
    pred_low = latest['pred_low']
    confidence = latest['confidence']
    
    # Solo ejecutar si confianza > 60%
    if confidence < 60:
        print(f"⚠️ Confianza baja ({confidence:.1f}%) - No se ejecuta")
        return
    
    # Calcular volumen (ejemplo: 0.01 ETH fijo, ajustar según tu capital)
    volume = 0.01
    
    # Calcular TP y SL
    side = signal.lower()
    tp, sl = calculate_tp_sl(current_price, side, atr, pred_high, pred_low, tp_percentage=0.80)
    
    print(f"\n{'='*70}")
    print(f"🚀 EJECUTANDO ORDEN")
    print(f"{'='*70}")
    print(f"📊 Señal: {signal}")
    print(f"💰 Precio: ${current_price:.2f}")
    print(f"📈 Volumen: {volume} ETH")
    print(f"🎯 TP: ${tp:.2f} ({((tp-current_price)/current_price*100):+.2f}%)")
    print(f"🛑 SL: ${sl:.2f} ({((sl-current_price)/current_price*100):+.2f}%)")
    print(f"{'='*70}\n")
    
    # Colocar orden en Kraken (market order)
    result = place_order(side, volume, None, tp, sl)
    
    if 'result' in result and 'txid' in result['result']:
        txid = result['result']['txid'][0]
        print(f"✅ Orden ejecutada: {txid}")
        
        # Guardar orden abierta
        order_data = {
            'txid': txid,
            'side': side,
            'entry_price': current_price,
            'volume': volume,
            'tp': tp,
            'sl': sl,
            'open_time': datetime.now().isoformat(),
            'signal_confidence': confidence
        }
        
        orders = [order_data]
        with open(OPEN_ORDERS_FILE, 'w') as f:
            json.dump(orders, f, indent=2)
        
        # Registro en CSV
        trade_data = {
            'timestamp': datetime.now(),
            'txid': txid,
            'side': side,
            'entry_price': current_price,
            'volume': volume,
            'tp': tp,
            'sl': sl,
            'confidence': confidence,
            'order_executed': 'YES',
            'order_type': signal
        }
        
        df = pd.DataFrame([trade_data])
        exec_file = 'orders_executed.csv'
        if os.path.exists(exec_file):
            df.to_csv(exec_file, mode='a', header=False, index=False)
        else:
            df.to_csv(exec_file, index=False)
        
        # Telegram
        msg = f"""
🚀 *Nueva Orden Ejecutada*

📊 Tipo: {signal}
💰 Precio Entrada: ${current_price:.2f}
📈 Volumen: {volume} ETH
🎯 TP: ${tp:.2f}
🛑 SL: ${sl:.2f}
📊 Confianza: {confidence:.1f}%

🔮 Predicción:
   HIGH: ${pred_high:.2f}
   LOW: ${pred_low:.2f}
"""
        send_telegram(msg)
        
    else:
        error = result.get('error', 'Unknown error')
        print(f"❌ Error al ejecutar orden: {error}")
        send_telegram(f"❌ Error ejecutando orden: {error}")

def main():
    print("="*70)
    print("  🤖 KRAKEN TRADER BOT")
    print("="*70)
    
    # 1. Monitorear órdenes existentes
    print("\n🔍 Monitoreando órdenes abiertas...")
    monitor_orders()
    
    # 2. Verificar nueva señal y ejecutar si corresponde
    print("\n📊 Verificando nuevas señales...")
    execute_signal()
    
    # 3. Mostrar resumen
    if os.path.exists(TRADES_FILE):
        df = pd.read_csv(TRADES_FILE)
        total_pnl = df['pnl_usd'].sum()
        win_rate = (df['pnl_usd'] > 0).sum() / len(df) * 100
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE TRADING")
        print(f"{'='*70}")
        print(f"Total trades: {len(df)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"P&L total: ${total_pnl:.2f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise

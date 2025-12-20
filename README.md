# ? ETH/USD LSTM Models - Auto-updated

![Last Update](https://img.shields.io/badge/Last%20Update-2025-12-20-blue)
![Status](https://img.shields.io/badge/Status-Active-green)

## ? Modelo ETH/USD LSTM

Modelo LSTM para predicci?n de precios ETH/USD, entrenado con datos hist?ricos de 1 hora.

### ? Estructura del Repositorio

```
ethusd-models/
??? models/
?   ??? pytorch/
?   ?   ??? ethusd_lstm_1h.pth            # Modelo PyTorch
?   ??? scalers/
?   ?   ??? ethusd_scaler_input_1h.pkl    # Scaler entrada
?   ?   ??? ethusd_scaler_output_1h.pkl   # Scaler salida
?   ??? config/
?       ??? ethusd_config_1h.json         # Configuraci?n
??? data/
?   ??? ETHUSD_1h_data.csv                # Dataset actualizado
??? results/
    ??? ethusd_results.png                # Gr?ficas de entrenamiento
```

### ? Actualizaci?n Autom?tica

- **Frecuencia:** Lunes, Mi?rcoles, S?bado a las 13:00 CET
- **M?todo:** Google Apps Script + Colab
- **?ltima actualizaci?n:** 2025-12-20

### ? Caracter?sticas del Modelo

- **Arquitectura:** LSTM
- **Par:** ETH/USD
- **Timeframe:** 1 hora
- **Formato:** PyTorch (.pth)

---

**? Auto-generado por Colab Automation**  
**? 20/12/2025, 13:52:29

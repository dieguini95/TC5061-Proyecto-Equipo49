"""Variables causales para P1: salidas por estación-hora, emitidas a medianoche.

El panel conserva horas sin demanda, pero no inventa ceros antes de conocer una
estación. La primera salida observada es una aproximación a su disponibilidad,
no evidencia de su fecha real de apertura ni de cierres posteriores.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_ml import config
from mibici_ml.config import (
    END_MONTH,
    P1_PANEL_PATH,
    START_MONTH,
    TRIPS_PROCESSED_DIR,
)

app = typer.Typer()


def validate_p1_forecast_contract() -> None:
    """Rechaza configuraciones que el pronóstico diario todavía no soporta.

    La emisión es a medianoche para las 24 horas del día. Permitir un horizonte
    menor sin cambiar ese panel admitiría rezagos del mismo día pronosticado.
    Se valida antes de leer datos y antes de registrar el protocolo en MLflow.
    """
    if config.P1_FORECAST_HORIZON_HOURS != 24 or config.P1_FORECAST_ORIGIN_HOUR != 0:
        raise ValueError("P1 soporta únicamente horizonte de 24 h y emisión a medianoche (0 h).")
    if not isinstance(config.P1_BASELINE_MEAN_WEEKS, int) or config.P1_BASELINE_MEAN_WEEKS < 1:
        raise ValueError("El número de semanas de P1 debe ser un entero positivo.")
    weekly_lag = config.P1_BASELINE_SEASONAL_LAG_HOURS
    lags = [*config.P1_LAG_HOURS, weekly_lag]
    if any(not isinstance(lag, int) or lag < 24 for lag in lags):
        raise ValueError(
            "Una previsión diaria a medianoche requiere rezagos enteros de al menos 24 h."
        )


def _months(start_month: str, end_month: str) -> pd.PeriodIndex:
    """Valida la ventana inclusiva de archivos mensuales."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month debe ser igual o posterior a start_month")
    return pd.period_range(start, end, freq="M")


def aggregate_departures(
    trips_dir: Path,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
) -> pd.DataFrame:
    """Cuenta salidas por estación y hora leyendo un Parquet mensual cada vez.

    Solo carga ``origin_id`` y ``start_time`` para limitar la memoria utilizada
    por millones de viajes. Cada archivo debe llamarse ``AAAA_MM.parquet`` y
    contener viajes del mes indicado, ya limpios por ``dataset.py``. Falla si
    falta un mes: la ausencia de archivos nunca se interpreta como demanda cero.
    Esta comprobación no demuestra cobertura operativa dentro de cada archivo.
    """
    months = _months(start_month, end_month)
    paths = [Path(trips_dir) / f"{month.year:04d}_{month.month:02d}.parquet" for month in months]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan meses de viajes: " + ", ".join(missing))

    monthly_counts = []
    for month, path in zip(months, paths, strict=True):
        trips = pd.read_parquet(path, columns=["origin_id", "start_time"])
        if trips.empty:
            raise ValueError(f"El archivo mensual no contiene viajes: {path}")
        timestamps = pd.to_datetime(trips["start_time"], errors="raise")
        if timestamps.dt.tz is not None:
            raise ValueError("start_time debe ser hora local sin zona horaria")
        if timestamps.isna().any() or not timestamps.dt.to_period("M").eq(month).all():
            raise ValueError(f"Fechas ausentes o fuera del mes {month}: {path}")
        station_ids = pd.to_numeric(trips["origin_id"], errors="raise")
        if (
            station_ids.isna().any()
            or not np.isfinite(station_ids).all()
            or not station_ids.mod(1).eq(0).all()
            or not station_ids.gt(0).all()
        ):
            raise ValueError(f"Identificadores de estación inválidos: {path}")
        departures = pd.DataFrame(
            {
                "station_id": station_ids.astype("int32"),
                "timestamp": timestamps.dt.floor("h"),
            }
        )
        counts = departures.groupby(["station_id", "timestamp"], sort=False).size()
        monthly_counts.append(counts.rename("departures").reset_index())
        logger.info("P1: {} viajes agregados de {}", len(trips), month)

    result = pd.concat(monthly_counts, ignore_index=True)
    result["departures"] = result["departures"].astype("int32")
    return result.sort_values(["station_id", "timestamp"], ignore_index=True)


def complete_station_hour_panel(
    counts: pd.DataFrame,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
) -> pd.DataFrame:
    """Completa un índice horario continuo y conserva la incertidumbre inicial.

    ``counts`` contiene salidas positivas por estación-hora. La estación pasa a
    ser elegible a la medianoche siguiente a su primera salida observada. Todo
    su primer día (posiblemente incompleto) y los anteriores quedan sin etiqueta;
    las horas posteriores sin viajes son ceros. ``eligible`` permite excluir del
    universo de predicción las estaciones todavía desconocidas en cada emisión.
    Los identificadores futuros solo existen en filas de almacenamiento no
    elegibles; no deben usarse para entrenar ni para definir categorías futuras.
    """
    validate_p1_forecast_contract()
    months = _months(start_month, end_month)
    required = {"station_id", "timestamp", "departures"}
    if not required.issubset(counts.columns):
        raise ValueError(f"Faltan columnas de conteos: {sorted(required - set(counts.columns))}")
    if counts.empty:
        raise ValueError("No hay salidas con las que construir el panel")
    counts = counts.loc[:, ["station_id", "timestamp", "departures"]].copy()
    counts["timestamp"] = pd.to_datetime(counts["timestamp"], errors="raise")
    if counts["timestamp"].dt.tz is not None:
        raise ValueError("timestamp debe ser hora local sin zona horaria")
    if counts[["station_id", "timestamp", "departures"]].isna().any().any():
        raise ValueError("Los conteos de entrada no pueden contener valores ausentes")
    if not counts["timestamp"].eq(counts["timestamp"].dt.floor("h")).all():
        raise ValueError("Los conteos deben estar agregados a horas exactas")
    if counts.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("Hay estaciones-hora duplicadas en los conteos")
    if not (counts["departures"].gt(0) & counts["departures"].mod(1).eq(0)).all():
        raise ValueError("Los conteos dispersos deben ser enteros positivos")

    start = months[0].start_time
    end_exclusive = (months[-1] + 1).start_time
    if not counts["timestamp"].between(start, end_exclusive, inclusive="left").all():
        raise ValueError("Hay conteos fuera de la ventana solicitada")
    hours = pd.date_range(start, end_exclusive, freq="h", inclusive="left")
    first_seen = counts.groupby("station_id")["timestamp"].min().sort_index()
    index = pd.MultiIndex.from_product(
        [first_seen.index, hours], names=["station_id", "timestamp"]
    )
    values = counts.set_index(["station_id", "timestamp"])["departures"]
    panel = values.reindex(index).fillna(0).astype("Int32").reset_index()
    panel["station_id"] = panel["station_id"].astype("int32")
    panel["first_seen"] = np.repeat(first_seen.to_numpy(), len(hours))
    panel["forecast_origin"] = panel["timestamp"].dt.normalize()
    panel["eligible"] = panel["first_seen"].lt(panel["forecast_origin"])
    panel.loc[~panel["eligible"], "departures"] = pd.NA
    return panel


def add_p1_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Añade rezagos y calendarios disponibles al emitir 24 horas a las 00:00.

    Verifica el índice horario antes de desplazar filas: cada rezago representa
    horas reales, incluso en días sin viajes. Los rezagos mínimos de 24 horas
    impiden utilizar salidas del día que se pronostica. Las referencias son la
    misma hora de la semana anterior y el promedio de las cuatro semanas previas.
    ``mean_4weeks_coverage`` explicita cuántas semanas existen; ``mean_4weeks``
    queda ausente hasta tener las cuatro, sin sustituciones silenciosas.
    ``evaluation_eligible`` identifica el mismo conjunto con cuatro semanas de
    historia para comparar todos los modelos. Las estaciones nuevas se reportan
    como cobertura excluida y requieren una estrategia futura de arranque.
    """
    validate_p1_forecast_contract()
    required = {"station_id", "timestamp", "departures", "first_seen"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Faltan columnas del panel: {sorted(required - set(panel.columns))}")
    if panel.empty:
        raise ValueError("El panel está vacío")
    result = panel.sort_values(["station_id", "timestamp"], ignore_index=True).copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    if result["timestamp"].dt.tz is not None:
        raise ValueError("timestamp debe ser hora local sin zona horaria")
    if (
        result["timestamp"].isna().any()
        or not result["timestamp"].eq(result["timestamp"].dt.floor("h")).all()
    ):
        raise ValueError("El panel debe contener horas exactas sin fechas ausentes")
    differences = result.groupby("station_id", sort=False)["timestamp"].diff()
    if not differences.dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("El índice horario debe ser continuo, sin huecos ni duplicados")

    result["forecast_origin"] = result["timestamp"].dt.normalize()
    result["eligible"] = result["first_seen"].lt(result["forecast_origin"])
    result.loc[~result["eligible"], "departures"] = pd.NA
    result["hour"] = result["timestamp"].dt.hour.astype("uint8")
    result["day_of_week"] = result["timestamp"].dt.dayofweek.astype("uint8")
    result["is_weekend"] = result["day_of_week"].ge(5)

    week_lags = [
        config.P1_BASELINE_SEASONAL_LAG_HOURS * week
        for week in range(1, config.P1_BASELINE_MEAN_WEEKS + 1)
    ]
    all_lags = sorted(set(config.P1_LAG_HOURS) | set(week_lags))
    grouped = result.groupby("station_id", sort=False)["departures"]
    shifted = {lag: grouped.shift(lag).astype("float32") for lag in all_lags}
    for lag in config.P1_LAG_HOURS:
        result[f"lag_{lag}"] = shifted[lag]
    result["seasonal_168h"] = shifted[config.P1_BASELINE_SEASONAL_LAG_HOURS]
    weeks = pd.concat([shifted[lag] for lag in week_lags], axis=1)
    result["mean_4weeks_coverage"] = weeks.notna().sum(axis=1).astype("uint8")
    full_history = result["mean_4weeks_coverage"].eq(config.P1_BASELINE_MEAN_WEEKS)
    result["mean_4weeks"] = weeks.mean(axis=1).where(full_history).astype("float32")
    result["evaluation_eligible"] = (
        result["eligible"]
        & result["departures"].notna()
        & full_history
        & result[[f"lag_{lag}" for lag in config.P1_LAG_HOURS]].notna().all(axis=1)
    )
    return result


def build_station_hour_panel(
    trips_dir: Path = TRIPS_PROCESSED_DIR,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
) -> pd.DataFrame:
    """Construye P1 desde meses completos, conservando trazabilidad temporal.

    El resultado incluye etiquetas, calendario, rezagos, dos baselines y máscaras
    de cobertura. No utiliza datos personales ni características de inventario.
    """
    validate_p1_forecast_contract()
    counts = aggregate_departures(trips_dir, start_month, end_month)
    panel = complete_station_hour_panel(counts, start_month, end_month)
    return add_p1_features(panel)


@app.command()
def main(
    trips_dir: Path = TRIPS_PROCESSED_DIR,
    output_path: Path = P1_PANEL_PATH,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
):
    """Genera el panel reproducible de P1 desde la terminal."""
    panel = build_station_hour_panel(trips_dir, start_month, end_month)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    logger.success(
        "Panel P1 guardado: {} filas, {} elegibles para evaluación, {}",
        len(panel),
        int(panel["evaluation_eligible"].sum()),
        output_path,
    )


if __name__ == "__main__":
    app()


# ---------------------------------------------------------------------------
# P2: riesgo de que una estacion quede vacia o llena en las siguientes H horas.
# ---------------------------------------------------------------------------


def build_p2_hourly_flows(trips: pd.DataFrame) -> pd.DataFrame:
    """Salidas y llegadas por estacion-hora a partir de viajes limpios.

    ``trips`` debe traer ``origin_id``, ``destination_id``, ``start_time`` y
    ``end_time`` tal como los produce ``dataset.py``. Las llegadas usan la hora
    de fin del viaje, por eso el inventario se actualiza al terminar el viaje.
    """
    required = {"origin_id", "destination_id", "start_time", "end_time"}
    if not required.issubset(trips.columns):
        raise ValueError(f"Faltan columnas de viajes: {sorted(required - set(trips.columns))}")
    departures = (
        trips.assign(
            est=trips["origin_id"].astype("int64"),
            ts=pd.to_datetime(trips["start_time"]).dt.floor("h"),
        )
        .groupby(["est", "ts"])
        .size()
        .rename("salidas")
    )
    arrivals = (
        trips.assign(
            est=trips["destination_id"].astype("int64"),
            ts=pd.to_datetime(trips["end_time"]).dt.floor("h"),
        )
        .groupby(["est", "ts"])
        .size()
        .rename("llegadas")
    )
    flows = pd.concat([departures, arrivals], axis=1).fillna(0).astype("int64")
    return flows.reset_index()


def build_p2_station_day_panel(
    flows: pd.DataFrame, capacities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel estacion-hora con flujo acumulado, rango factible y etiqueta de riesgo.

    Para cada estacion-dia se calcula el flujo neto acumulado ``F(t)`` y el rango
    de inventario inicial ``[L, U]`` que mantiene el inventario entre 0 y la
    capacidad ``C``: ``L = max(0, -min F)``, ``U = min(C, C - max F)``. El dia es
    factible si ``L <= U``. Los dias infactibles se descartan porque con solo
    viajes no hay un inventario consistente.

    Devuelve ``(panel, bounds)``: ``panel`` contiene solo dias factibles con
    capacidad positiva; ``bounds`` resume cada estacion-dia (incluidos los
    infactibles) para el registro de calidad.
    """
    if not {"est", "capacidad"}.issubset(capacities.columns):
        raise ValueError("capacities debe contener las columnas est y capacidad")
    if flows.empty:
        raise ValueError("No hay flujos para construir el panel de P2")
    stations = capacities.loc[capacities["capacidad"] > 0, ["est", "capacidad"]]
    grid_hours = pd.date_range(
        flows["ts"].min().normalize(),
        flows["ts"].max().floor("D") + pd.Timedelta(hours=23),
        freq="h",
    )
    full_index = pd.MultiIndex.from_product([stations["est"], grid_hours], names=["est", "ts"])
    panel = (
        flows.set_index(["est", "ts"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .merge(stations, on="est", how="inner")
    )
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["day"] = panel["ts"].dt.normalize()
    panel = panel.sort_values(["est", "ts"]).reset_index(drop=True)
    panel["flujo_acumulado"] = panel.groupby(["est", "day"])["neto"].cumsum()

    bounds = panel.groupby(["est", "day"], sort=False).agg(
        max_f=("flujo_acumulado", "max"),
        min_f=("flujo_acumulado", "min"),
        capacidad=("capacidad", "first"),
    )
    bounds["max_f"] = bounds["max_f"].clip(lower=0)
    bounds["min_f"] = bounds["min_f"].clip(upper=0)
    bounds["lower"] = np.maximum(0, -bounds["min_f"])
    bounds["upper"] = np.minimum(bounds["capacidad"], bounds["capacidad"] - bounds["max_f"])
    bounds["factible"] = bounds["lower"] <= bounds["upper"]
    bounds["s0"] = (bounds["lower"] + bounds["upper"]) / 2
    bounds = bounds.reset_index()

    panel = panel.merge(bounds[["est", "day", "factible", "s0"]], on=["est", "day"], how="left")
    panel["factible"] = panel["factible"].eq(True)
    panel = panel.loc[panel["factible"]].copy()
    panel["inventario"] = panel["s0"] + panel["flujo_acumulado"]
    panel["label"] = p2_risk_labels(panel, config.P2_HORIZON_HOURS)
    return panel.reset_index(drop=True), bounds


def p2_risk_labels(panel: pd.DataFrame, horizon: int) -> np.ndarray:
    """Etiqueta ``r(s,t)=1`` si el inventario estimado toca 0 o C en las siguientes H horas.

    La etiqueta puede mirar al futuro porque es la respuesta que se quiere
    predecir; las features nunca lo hacen. Las ventanas no cruzan de dia, porque
    el inventario inicial ``s0`` se estima por dia de servicio.
    """
    empty = (panel["inventario"] <= 0).to_numpy()
    full = (panel["inventario"] >= panel["capacidad"]).to_numpy()
    risk = (empty | full).astype("int64")
    frame = pd.DataFrame(
        {"est": panel["est"].to_numpy(), "day": panel["day"].to_numpy(), "risk": risk}
    )
    labels = np.zeros(len(panel), dtype="int64")
    for step in range(1, horizon + 1):
        future = frame.groupby(["est", "day"])["risk"].shift(-step)
        labels = np.maximum(labels, future.fillna(0).to_numpy().astype("int64"))
    return labels


def build_p2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Variables de P2 que solo usan informacion hasta el instante ``t``.

    Incluye hora y calendario, capacidad, flujo acumulado ``F(t)`` y su proporcion
    de la capacidad, llegadas y salidas de la ultima y las dos ultimas horas, y
    salidas de la misma hora del dia anterior. Los rezagos se calculan con
    ``shift`` hacia el pasado, por lo que ninguna columna mira a ``t+1`` o mas.
    """
    required = {"est", "ts", "day", "salidas", "llegadas", "capacidad", "flujo_acumulado"}
    if not required.issubset(panel.columns):
        raise ValueError(
            f"Faltan columnas del panel de P2: {sorted(required - set(panel.columns))}"
        )
    frame = panel[
        ["est", "ts", "day", "salidas", "llegadas", "capacidad", "flujo_acumulado"]
    ].copy()
    grouped = frame.groupby("est", sort=False)
    frame["hora"] = frame["ts"].dt.hour
    frame["dia_semana"] = frame["ts"].dt.dayofweek
    frame["fin_de_semana"] = (frame["dia_semana"] >= 5).astype("int64")
    frame["flujo_acumulado_sobre_c"] = frame["flujo_acumulado"] / frame["capacidad"]
    frame["llegadas_1h"] = frame["llegadas"]
    frame["salidas_1h"] = frame["salidas"]
    frame["llegadas_2h"] = frame["llegadas"] + grouped["llegadas"].shift(1).fillna(0)
    frame["salidas_2h"] = frame["salidas"] + grouped["salidas"].shift(1).fillna(0)
    frame["salidas_misma_hora_ayer"] = grouped["salidas"].shift(24).fillna(0)
    return frame.reset_index(drop=True)


def add_p2_v2_features(panel_features: pd.DataFrame) -> pd.DataFrame:
    """Features v2 de P2: estado observado hasta ``t`` (sin mirar a ``t+1`` o mas).

    - ``riesgo_observado_h``: 1 si el inventario estimado en ``t`` ya estaba en 0 o en C.
      Es el estado que hoy observa la persistencia, pero como variable del modelo.
    - ``riesgo_observado_previo_h``: el mismo estado ``H`` horas antes de ``t``.
    - ``tendencia_flujo_3h``: cambio de ``F(t)`` en las ultimas 3 horas, sobre capacidad.

    ``inventario`` y ``capacidad`` vienen del panel; el inventario de ``t`` solo usa
    flujos hasta ``t``, y los rezagos se calculan con ``shift`` hacia el pasado.
    """
    required = {"est", "day", "flujo_acumulado", "capacidad"}
    if not required.issubset(panel_features.columns):
        raise ValueError(
            f"Faltan columnas para features v2: {sorted(required - set(panel_features.columns))}"
        )
    frame = panel_features.copy()
    grouped = frame.groupby(["est", "day"], sort=False)
    # Inventario estimado hasta t: rango inicial s0 mas el flujo acumulado observado.
    inventory_now = frame["flujo_acumulado"]
    if "inventario" in frame.columns:
        inventory_now = frame["inventario"]
    empty_or_full = ((inventory_now <= 0) | (inventory_now >= frame["capacidad"])).astype("int64")
    frame["riesgo_observado_h"] = empty_or_full
    frame["riesgo_observado_previo_h"] = (
        frame.groupby(["est", "day"], sort=False)["riesgo_observado_h"]
        .shift(config.P2_HORIZON_HOURS)
        .fillna(0)
        .astype("int64")
    )
    frame["tendencia_flujo_3h"] = (
        frame["flujo_acumulado"] - grouped["flujo_acumulado"].shift(3).fillna(0)
    ) / frame["capacidad"]
    return frame


P2_FEATURE_COLUMNS = (
    "hora",
    "dia_semana",
    "fin_de_semana",
    "capacidad",
    "flujo_acumulado",
    "flujo_acumulado_sobre_c",
    "llegadas_1h",
    "salidas_1h",
    "llegadas_2h",
    "salidas_2h",
    "salidas_misma_hora_ayer",
)

P2_FEATURE_COLUMNS_V2 = P2_FEATURE_COLUMNS + (
    "riesgo_observado_h",
    "riesgo_observado_previo_h",
    "tendencia_flujo_3h",
)

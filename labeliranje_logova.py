from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "firewall_logs_unlabeled"
OUTPUT_DIR = BASE_DIR / "firewall_logs_labeled"

TIME_WINDOW = "1min"

NORMAL_LABEL = "Normal"
PORT_SCAN_LABEL = "Port Scan"
BRUTE_FORCE_LABEL = "Brute Force"
DDOS_LABEL = "DDoS"

PORT_SCAN_MIN_UNIQUE_PORTS = 25
BRUTE_FORCE_MIN_ATTEMPTS = 15
DDOS_MIN_EVENTS = 80
DDOS_MIN_SOURCES = 8

BRUTE_FORCE_PORTS = {22, 3389}
DDOS_PORTS = {80, 443, 8080}


def output_name(input_path: Path) -> str:
    if "nonlabeled" in input_path.name:
        return input_path.name.replace("nonlabeled", "labeled")
    return f"{input_path.stem}_labeled{input_path.suffix}"


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["time_window"] = df["timestamp"].dt.floor(TIME_WINDOW)

    for column in ["src_port", "dst_port", "bytes", "packets"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    for column in [
        "action",
        "src_zone",
        "dst_zone",
        "protocol",
        "tcp_flags",
        "session_status",
        "session_reason",
    ]:
        df[column] = df[column].astype(str)

    df["label"] = NORMAL_LABEL
    return df


def label_port_scan(df: pd.DataFrame) -> None:
    probe_like = (
        df["protocol"].isin(["TCP", "UDP"])
        & (df["src_port"] >= 1024)
        & (
            df["protocol"].eq("UDP")
            | df["tcp_flags"].eq("S")
            | df["action"].isin(["deny", "drop"])
        )
    )

    scan_counts = (
        df.loc[probe_like]
        .groupby(["src_ip", "time_window"], dropna=False)
        .agg(unique_ports=("dst_port", "nunique"), unique_targets=("dst_ip", "nunique"))
        .reset_index()
    )

    scan_windows = scan_counts[
        scan_counts["unique_ports"] >= PORT_SCAN_MIN_UNIQUE_PORTS
    ][["src_ip", "time_window"]]

    for _, scan in scan_windows.iterrows():
        scanner_ip = scan["src_ip"]
        window = scan["time_window"]
        cond = (
            df["time_window"].eq(window)
            & df["protocol"].isin(["TCP", "UDP"])
            & (df["src_ip"].eq(scanner_ip) | df["dst_ip"].eq(scanner_ip))
        )
        df.loc[cond, "label"] = PORT_SCAN_LABEL


def label_brute_force(df: pd.DataFrame) -> None:
    login_like = (
        df["protocol"].eq("TCP")
        & df["dst_port"].isin(BRUTE_FORCE_PORTS)
        & df["tcp_flags"].isin(["S", "PA", "FA"])
        & (df["src_port"] >= 1024)
    )

    brute_counts = (
        df.loc[login_like]
        .groupby(["src_ip", "dst_ip", "dst_port", "time_window"], dropna=False)
        .size()
        .reset_index(name="attempts")
    )

    brute_windows = brute_counts[
        brute_counts["attempts"] >= BRUTE_FORCE_MIN_ATTEMPTS
    ][["src_ip", "dst_ip", "dst_port", "time_window"]]

    for _, brute in brute_windows.iterrows():
        cond = (
            df["src_ip"].eq(brute["src_ip"])
            & df["dst_ip"].eq(brute["dst_ip"])
            & df["dst_port"].eq(brute["dst_port"])
            & df["time_window"].eq(brute["time_window"])
            & df["protocol"].eq("TCP")
        )
        df.loc[cond, "label"] = BRUTE_FORCE_LABEL


def label_ddos(df: pd.DataFrame) -> None:
    ddos_like = (
        df["protocol"].isin(["TCP", "UDP"])
        & df["dst_zone"].eq("SERVER")
        & df["src_zone"].eq("WAN")
        & df["dst_port"].isin(DDOS_PORTS)
        & (df["src_port"] >= 1024)
    )

    ddos_counts = (
        df.loc[ddos_like]
        .groupby(["dst_ip", "time_window"], dropna=False)
        .agg(
            events=("src_ip", "size"),
            sources=("src_ip", "nunique"),
            ports=("dst_port", "nunique"),
        )
        .reset_index()
    )

    ddos_windows = ddos_counts[
        (ddos_counts["events"] >= DDOS_MIN_EVENTS)
        & (ddos_counts["sources"] >= DDOS_MIN_SOURCES)
    ][["dst_ip", "time_window"]]

    for _, ddos in ddos_windows.iterrows():
        cond = (
            df["dst_ip"].eq(ddos["dst_ip"])
            & df["time_window"].eq(ddos["time_window"])
            & df["protocol"].isin(["TCP", "UDP"])
            & df["src_zone"].eq("WAN")
            & df["dst_zone"].eq("SERVER")
            & df["dst_port"].isin(DDOS_PORTS)
            & (df["src_port"] >= 1024)
        )
        df.loc[cond, "label"] = DDOS_LABEL


def label_file(input_path: Path) -> Path:
    df = pd.read_csv(input_path)
    df = prepare_columns(df)

    label_port_scan(df)
    label_brute_force(df)
    label_ddos(df)

    df = df.drop(columns=["time_window"])
    output_path = OUTPUT_DIR / output_name(input_path)
    df.to_csv(output_path, index=False)

    counts = df["label"].value_counts().to_dict()
    print(f"{input_path.name} -> {output_path.name} | {counts}")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nema CSV datoteka u folderu: {INPUT_DIR}")

    for csv_file in csv_files:
        label_file(csv_file)


if __name__ == "__main__":
    main()

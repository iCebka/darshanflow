#!/usr/bin/env python3
"""
Expected structure *_ops.json:
    {
      "MODULE": [
        {"file": "...", "id": ..., "rank": ..., "counters": {...}, "fcounters": {...}},
        ...
      ],
      ...
      "DXT_POSIX": {"info": "Not implemented."}   # módulos no-lista se ignoran
    }

Use:
    python3 aggregate_ops.py mc_normalized_rntuple_1M.h5
    python3 aggregate_ops.py .h5
    python3 aggregate_ops.py .h5 --dir /ruta/a/los/ops_json
    python3 aggregate_ops.py .h5 --json acumulado_h5.json
    python3 aggregate_ops.py mc_normalized_rntuple_1M.h5 --exact
"""

import json
import sys
import argparse
import glob
import os
from collections import defaultdict, OrderedDict


def find_ops_files(directory):
    pattern = os.path.join(directory, "*_ops.json")
    return sorted(glob.glob(pattern))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def matches(file_field, pattern, exact=False):
    if not file_field:
        return False
    if exact:
        return file_field == pattern or os.path.basename(file_field) == pattern
    return pattern in file_field


def aggregate(ops_files, pattern, exact=False):
    """
    Devuelve un dict:
      {
        module: {
          "counters": {name: suma},
          "fcounters": {name: suma},
          "n_records": int,
          "matched_paths": set(rutas exactas que hicieron match),
          "ranks": set(ranks involucrados),
          "source_files": set(nombres de los *_ops.json de origen),
          "records": [ {source_file, file, rank, id}, ... ]  # trazabilidad
        },
        ...
      }
    """
    result = defaultdict(lambda: {
        "counters": defaultdict(float),
        "fcounters": defaultdict(float),
        "n_records": 0,
        "matched_paths": set(),
        "ranks": set(),
        "source_files": set(),
        "records": [],
    })

    for path in ops_files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"AVISO: no se pudo leer {path}: {e}", file=sys.stderr)
            continue

        base = os.path.basename(path)

        for module, recs in data.items():
            if not isinstance(recs, list):
                # p.ej. DXT_POSIX = {"info": "Not implemented."}
                continue

            for rec in recs:
                file_field = rec.get("file")
                if not matches(file_field, pattern, exact=exact):
                    continue

                mod_acc = result[module]
                mod_acc["n_records"] += 1
                mod_acc["matched_paths"].add(file_field)
                mod_acc["ranks"].add(rec.get("rank"))
                mod_acc["source_files"].add(base)

                for k, v in rec.get("counters", {}).items():
                    if isinstance(v, (int, float)):
                        mod_acc["counters"][k] += v

                for k, v in rec.get("fcounters", {}).items():
                    if isinstance(v, (int, float)):
                        mod_acc["fcounters"][k] += v

                mod_acc["records"].append({
                    "source_file": base,
                    "file": file_field,
                    "rank": rec.get("rank"),
                    "id": rec.get("id"),
                })

    return result


def fmt_num(v):
    if float(v).is_integer():
        return str(int(v))
    return str(v)


def print_summary(result, pattern, n_ops_files):
    print("=" * 80)
    print(f"REPORTE ACUMULADO - archivos que coinciden con: '{pattern}'")
    print(f"(revisados {n_ops_files} archivos *_ops.json)")
    print("=" * 80)

    if not result:
        print("\nNo se encontraron registros que coincidan con el patrón dado.")
        return

    for module, acc in result.items():
        print(f"\n--- Módulo: {module} ---")
        print(f"  Registros acumulados : {acc['n_records']}")
        print(f"  Archivos *_ops.json  : {len(acc['source_files'])}")
        print(f"  Ranks involucrados   : {sorted(r for r in acc['ranks'] if r is not None)}")
        print(f"  Rutas que hicieron match:")
        for p in sorted(acc["matched_paths"]):
            print(f"    - {p}")

        print("\n  Counters acumulados (suma):")
        for k in sorted(acc["counters"]):
            v = acc["counters"][k]
            if v == 0:
                continue
            print(f"    {k:35s} = {fmt_num(v)}")

        print("\n  Fcounters acumulados (suma; ojo: sumar *_TIMESTAMP no tiene sentido físico,")
        print("  son marcas de tiempo absolutas, no duraciones. Los *_TIME sí son duraciones):")
        for k in sorted(acc["fcounters"]):
            v = acc["fcounters"][k]
            if v == 0:
                continue
            print(f"    {k:35s} = {v}")


def to_serializable(result):
    out = OrderedDict()
    for module, acc in result.items():
        out[module] = {
            "n_records": acc["n_records"],
            "n_source_files": len(acc["source_files"]),
            "ranks": sorted(r for r in acc["ranks"] if r is not None),
            "matched_paths": sorted(acc["matched_paths"]),
            "source_files": sorted(acc["source_files"]),
            "counters": dict(sorted(acc["counters"].items())),
            "fcounters": dict(sorted(acc["fcounters"].items())),
            "records": acc["records"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Acumula counters/fcounters de Darshan a través de múltiples *_ops.json, "
                    "filtrando registros cuyo campo 'file' coincida con un patrón."
    )
    parser.add_argument(
        "pattern",
        help="Nombre de archivo o substring a buscar en el campo 'file' "
             "(ej: 'mc_normalized_rntuple_1M.h5' o simplemente '.h5')",
    )
    parser.add_argument(
        "--dir", default=".",
        help="Directorio con los *_ops.json (por defecto: directorio actual)",
    )
    parser.add_argument(
        "--exact", action="store_true",
        help="Coincidencia exacta de ruta/basename en vez de substring",
    )
    parser.add_argument(
        "--json", metavar="OUTPUT",
        help="Guardar también el resultado acumulado en un archivo JSON",
    )
    args = parser.parse_args()

    ops_files = find_ops_files(args.dir)
    if not ops_files:
        print(f"No se encontraron archivos *_ops.json en {args.dir}", file=sys.stderr)
        sys.exit(1)

    result = aggregate(ops_files, args.pattern, exact=args.exact)
    print_summary(result, args.pattern, len(ops_files))

    if args.json:
        serializable = to_serializable(result)
        with open(args.json, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\nResultado acumulado guardado en: {args.json}")


if __name__ == "__main__":
    main()
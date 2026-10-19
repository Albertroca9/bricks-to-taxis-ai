"""Ejecuta el código del notebook en orden con fixtures; no produce resultados académicos.

Solo sustituye descarga, directorio de salida y tamaño de bosques. Mantiene los
36 candidatos, cinco folds, exportación, gráficos y controles del notebook.
"""
import ast
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'tmp' / 'lego_smoke'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')


def make_fixture():
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw
    WORK.mkdir(parents=True, exist_ok=True)
    images = WORK / 'dataset'
    images.mkdir(exist_ok=True)
    frame = pd.read_csv(ROOT / '01 - DCCubos/piezas_dataset.csv')
    frame = frame.groupby('label', group_keys=False).head(20).reset_index(drop=True)
    rng = np.random.default_rng(281)
    labels = sorted(frame.label.unique())
    for number, row in enumerate(frame.itertuples()):
        image = Image.new('L', (128, 128), 250)
        draw = ImageDraw.Draw(image)
        group = labels.index(row.label)
        left, top = 20 + int(rng.integers(0, 8)), 20 + int(rng.integers(0, 8))
        right = 82 + 4 * (group % 3)
        bottom = 70 + 5 * (group % 4)
        shade = 40 + int(rng.integers(0, 45))
        if group % 3 == 0:
            draw.ellipse((left, top, right, bottom), fill=shade)
        elif group % 3 == 1:
            draw.polygon([(left, bottom), (left + 10, top), (right, bottom)], fill=shade)
        else:
            draw.rectangle((left, top, right, bottom), fill=shade)
        # Pequeña marca interna por imagen: mantiene los fixtures sin copias exactas.
        draw.line([(40, 45), (45 + number % 15, 50 + number // 15)], fill=110, width=2)
        image.save(images / row.filename)
    frame.to_csv(WORK / 'piezas_dataset.csv', index=False)
    return images


def main():
    import nbformat
    images = make_fixture()
    notebook = nbformat.read(ROOT / '01 - DCCubos/Tarea1_LEGO.ipynb', as_version=4)
    nbformat.validate(notebook)
    namespace = {'__name__': '__main__'}
    checked = 0
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        tags = cell.metadata.get('tags', [])
        if 'installation' in tags:
            continue
        source = cell.source
        if 'configuration' in tags:
            source = source.replace('BASE_DIR = Path.cwd()', f'BASE_DIR = Path({str(WORK)!r})')
            source += f'\nLOCAL_IMAGES = Path({str(images)!r})\nN_JOBS = 1\n'
        if 'model_configuration' in tags:
            source = source.replace("'n_estimators': 200", "'n_estimators': 8")
        if 'report_inputs' in tags:
            source += "\nREPORT_AUTHOR = 'PRUEBA SINTÉTICA - NO ENTREGAR'\n"
        ast.parse(source)
        print(f'SMOKE cell {number}: {source.splitlines()[0][:85]}', flush=True)
        exec(compile(source, f'{notebook.cells[number].id}', 'exec'), namespace)
        if 'display' in namespace:
            namespace['display'] = lambda *args, **kwargs: None
        if 'plt' in namespace:
            namespace['plt'].show = lambda *args, **kwargs: None
            namespace['plt'].close('all')
        checked += 1
    results = namespace['RESULTS']
    assert len(namespace['ranking']) == 36
    assert len(namespace['final_comparison']) == 12
    assert len(namespace['category_comparison']) == 2
    assert (results / 'respuesta_borrador.pdf').read_bytes().startswith(b'%PDF')
    assert len(list((results / 'figuras').glob('10_matriz_top*.png'))) == 3
    import zipfile
    with zipfile.ZipFile(WORK / 'resultados_LEGO.zip') as archive:
        assert archive.testzip() is None
        assert not any('/dataset/' in name or '/cache/' in name for name in archive.namelist())
    (WORK / 'SMOKE_ONLY.txt').write_text('Datos artificiales: solo comprobación funcional, no resultados de LEGO.\n', encoding='utf-8')
    (WORK / 'latest_results.txt').write_text(str(results), encoding='utf-8')
    print(f'SMOKE OK: {checked} celdas, 36 candidatos, 12 finalistas y PDF/ZIP verificados. Salida: {WORK}', flush=True)


if __name__ == '__main__':
    main()

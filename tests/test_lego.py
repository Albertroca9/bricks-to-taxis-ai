"""Pruebas de contratos del notebook; no requieren descargar imágenes de Kaggle."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / '01 - DCCubos' / 'Tarea1_LEGO.ipynb'


def load_namespace():
    namespace = {'__name__': 'lego_notebook'}
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'definitions' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), namespace)
    return namespace


class LegoContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_namespace()

    def function(self, name):
        self.assertTrue(name in self.ns, f'Falta implementar {name} en el notebook')
        return self.ns[name]

    def test_join_uses_filename_and_rejects_missing_or_duplicate_keys(self):
        align = self.function('align_sources')
        pd = self.ns['pd']
        manifest = pd.DataFrame({'filename': ['b.png', 'a.png'], 'label': ['B', 'A']})
        inventory = pd.DataFrame({'filename': ['a.png', 'b.png'], 'label': ['A', 'B'], 'weight_g': [1., 2.]})
        joined = align(manifest, inventory)
        self.assertEqual(joined['weight_g'].tolist(), [2., 1.])
        with self.assertRaises(ValueError):
            align(manifest, inventory.iloc[:1])
        with self.assertRaises(ValueError):
            align(manifest, pd.concat([inventory, inventory.iloc[:1]]))

    def test_cleaning_is_label_independent_and_preserves_rows(self):
        clean = self.function('clean_inventory')
        pd = self.ns['pd']
        source = pd.read_csv(ROOT / '01 - DCCubos' / 'piezas_dataset.csv')
        result = clean(source)
        shuffled = source.copy()
        shuffled['label'] = 'NO_USAR'
        other = clean(shuffled)
        self.assertEqual(len(result), 2400)
        pd.testing.assert_frame_equal(result.drop(columns='label'), other.drop(columns='label'))
        self.assertTrue(result.n_studs.dropna().ge(0).all())
        self.assertTrue(result.height_mm.dropna().le(30).all())
        self.assertTrue(result.price.dropna().gt(0).all())
        row = source.iloc[[0]].copy()
        row['price'] = '$0,25'
        self.assertAlmostEqual(clean(row).price.iloc[0], .25)

    def test_split_is_reproducible_disjoint_and_stratified(self):
        split = self.function('make_partition')
        pd = self.ns['pd']
        data = pd.read_csv(ROOT / '01 - DCCubos' / 'piezas_dataset.csv')
        first = split(data, seed=42)
        second = split(data.sample(frac=1, random_state=9), seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first['split'].value_counts().to_dict(), {'train': 1920, 'test': 480})
        self.assertEqual(first.groupby(['label', 'split']).size().unstack()['test'].tolist(), [60] * 8)

    def test_empty_mask_fails_and_rectangle_features_are_finite(self):
        extract = self.function('extract_image_features')
        np = self.ns['np']
        blank = np.full((128, 128), 255, dtype=np.uint8)
        with self.assertRaises(ValueError):
            extract(blank)
        rectangle = blank.copy()
        rectangle[32:96, 24:104] = 50
        features, mask = extract(rectangle)
        self.assertEqual(len(features), 1770)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(mask.sum(), 4500)

    def test_cv_fits_imputer_without_validation_and_excludes_identifiers(self):
        build = self.function('build_pipeline')
        pd = self.ns['pd']
        np = self.ns['np']
        params = {'max_depth': 3, 'min_samples_leaf': 1}
        pipe = build('B', 'DecisionTree', params, visual_columns=[], seed=42)
        columns = self.ns['TABULAR_COLUMNS']
        data = pd.DataFrame({c: [1., 2., np.nan, 4., 5., 6.] for c in columns})
        data['filename'] = ['secret'] * 6
        data['category'] = ['leak'] * 6
        pipe.fit(data.iloc[:4], ['a', 'b', 'a', 'b'])
        medians = pipe.named_steps['preprocess'].named_transformers_['tabular'].named_steps['imputer'].statistics_
        np.testing.assert_allclose(medians, [2.] * 5)
        forbidden = {'filename', 'label', 'category'}
        self.assertFalse(forbidden & set(pipe.named_steps['preprocess'].transformers_[0][2]))

    def test_identical_images_stay_in_one_partition(self):
        split = self.function('make_partition')
        pd = self.ns['pd']
        data = pd.DataFrame([{'filename': f'{label}_{group}_{copy}.png', 'label': label,
                              'image_hash': f'{label}_{group}'}
                             for label in ['A', 'B'] for group in range(20) for copy in range(2)])
        result = split(data).merge(data[['filename', 'image_hash']], on='filename')
        self.assertTrue(result.groupby('image_hash')['split'].nunique().eq(1).all())

    def test_ranking_ignores_test_score_and_breaks_ties_deterministically(self):
        rank = self.function('rank_candidates')
        pd = self.ns['pd']
        result = rank(pd.DataFrame({'id': ['c', 'b', 'a'], 'cv_accuracy': [.8, .9, .9],
                                    'cv_std': [.01, .02, .02], 'test_accuracy': [1., .8, .7]}))
        self.assertEqual(result.id.tolist(), ['a', 'b', 'c'])

    def test_report_writes_pdf_and_markdown_with_unicode(self):
        render = self.function('render_report')
        import tempfile
        pd = self.ns['pd']
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            render([('heading', 'Prueba: clasificación'), ('paragraph', 'α, π y comparación de precisión.'),
                    ('table', ('Métricas', pd.DataFrame({'modelo': ['árbol'], 'accuracy': [.8]})))], base)
            self.assertTrue((base / 'respuesta_borrador.pdf').read_bytes().startswith(b'%PDF'))
            self.assertIn('clasificación', (base / 'respuesta_borrador.md').read_text(encoding='utf-8'))

    def test_anova_handles_constant_background_without_warnings(self):
        scoring = self.function('safe_f_classif')
        import warnings
        np = self.ns['np']
        values = np.array([[0, 1., 2.], [0, 2., 2.], [0, 4., 5.], [0, 5., 5.]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            scores, pvalues = scoring(values, np.array(['a', 'a', 'b', 'b']))
        self.assertEqual(len(caught), 0)
        self.assertEqual(scores[0], 0.)
        self.assertEqual(pvalues[0], 1.)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[2], scores[1])


if __name__ == '__main__':
    unittest.main()

export default function MetricsPanel({ metrics }) {
  if (!metrics) {
    return <div className="empty-state">Train the backend model or load metrics to see evaluation details.</div>;
  }

  const report = metrics.classification_report || {};
  const rows = ['High', 'Medium', 'Low']
    .filter((label) => report[label])
    .map((label) => ({
      label,
      precision: report[label].precision,
      recall: report[label].recall,
      f1: report[label]['f1-score'],
      support: report[label].support
    }));

  return (
    <div className="metrics-grid">
      <div className="mini-grid">
        <div className="mini-card"><span>Accuracy</span><strong>{metrics.accuracy ?? '-'}</strong></div>
        <div className="mini-card"><span>Macro F1</span><strong>{metrics.macro_f1 ?? '-'}</strong></div>
        <div className="mini-card"><span>Baseline Logistic F1</span><strong>{metrics.baseline_logistic_macro_f1 ?? '-'}</strong></div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Band</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{row.precision?.toFixed?.(3) ?? row.precision}</td>
                <td>{row.recall?.toFixed?.(3) ?? row.recall}</td>
                <td>{row.f1?.toFixed?.(3) ?? row.f1}</td>
                <td>{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="best-params">
        <h3>Best Random Forest Parameters</h3>
        <pre>{JSON.stringify(metrics.best_params || {}, null, 2)}</pre>
      </div>
    </div>
  );
}

"use client";

import { useMemo, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

type PredictionClass =
  | "alpha-amylase"
  | "isoamylase"
  | "pullulanase/limit dextrinase";

type PredictionResponse = {
  prediction: PredictionClass;
  probabilities: Record<PredictionClass, number>;
};

const CLASS_INFO: Record<
  PredictionClass,
  {
    title: string;
    shortTitle: string;
    description: string;
  }
> = {
  "alpha-amylase": {
    title: "α-Amylase",
    shortTitle: "α-Amylase",
    description:
      "Hydrolytic activity associated with internal α-1,4 glycosidic bonds.",
  },
  isoamylase: {
    title: "Isoamylase",
    shortTitle: "Isoamylase",
    description:
      "Debranching activity associated with α-1,6 linkages in starch structures.",
  },
  "pullulanase/limit dextrinase": {
    title: "Pullulanase / LDA",
    shortTitle: "Pullulanase / LDA",
    description:
      "Debranching activity associated with pullulan and limit dextrins.",
  },
};

const CLASS_ORDER: PredictionClass[] = [
  "alpha-amylase",
  "isoamylase",
  "pullulanase/limit dextrinase",
];

export default function Home() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const cleanSequence = useMemo(
    () => sequence.replace(/\s+/g, "").toUpperCase(),
    [sequence]
  );

  const confidence = result
    ? result.probabilities[result.prediction]
    : 0;

  const confidencePercent = confidence * 100;

  async function handlePredict() {
    setError("");
    setResult(null);

    if (!cleanSequence) {
      setError("Enter an amino-acid sequence.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sequence: cleanSequence,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data: PredictionResponse = await response.json();

      setResult(data);
    } catch {
      setError(
        "Unable to connect to the prediction server. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleExample() {
    setSequence(
      "MALTLTPTSSVHLLSSISVARPRIFAADFNLRSRWRRRRPVTSISN"
    );

    setResult(null);
    setError("");
  }

  function handleClear() {
    setSequence("");
    setResult(null);
    setError("");
  }

  return (
    <main className="min-h-screen bg-[#f7f8f4] text-[#172018]">
      <section className="mx-auto max-w-7xl px-6 py-10 lg:px-10">
        {/* HEADER */}
        <header className="mb-16 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold uppercase tracking-[0.25em] text-[#657465]">
              Plant Enzyme ML
            </div>

            <div className="mt-1 text-xs text-[#899289]">
              GH13 functional specificity prediction
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-[#d9dfd6] bg-white px-4 py-2 text-xs font-medium text-[#657465]">
            <span className="h-2 w-2 rounded-full bg-[#7c9976]" />
            ESM-2 · ML
          </div>
        </header>

        {/* HERO */}
        <section className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div>
            <div className="mb-5 inline-flex rounded-full bg-[#e8eee4] px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-[#506150]">
              Protein sequence analysis
            </div>

            <h1 className="max-w-3xl text-5xl font-semibold leading-[1.05] tracking-tight lg:text-7xl">
              Predict plant enzyme
              <span className="block text-[#71856e]">
                functional specificity.
              </span>
            </h1>

            <p className="mt-7 max-w-2xl text-lg leading-8 text-[#667066]">
              Submit an amino-acid sequence and classify a GH13 enzyme into
              α-amylase, isoamylase, or pullulanase / limit dextrinase.
            </p>
          </div>

          {/* PIPELINE */}
          <div className="rounded-[2rem] border border-[#dce2d9] bg-white p-7 shadow-[0_20px_60px_rgba(40,55,40,0.08)]">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold">
                  Prediction pipeline
                </div>

                <div className="mt-1 text-xs text-[#899289]">
                  Sequence → embedding → classifier
                </div>
              </div>

              <div className="h-3 w-3 rounded-full bg-[#7c9976]" />
            </div>

            <div className="space-y-3">
              {[
                ["01", "Amino-acid sequence"],
                ["02", "ESM-2 embedding"],
                ["03", "Logistic Regression"],
                ["04", "Functional class"],
              ].map(([number, label]) => (
                <div
                  key={number}
                  className="flex items-center gap-4 rounded-2xl bg-[#f5f7f3] px-4 py-4"
                >
                  <span className="font-mono text-xs text-[#8c978b]">
                    {number}
                  </span>

                  <span className="text-sm font-medium">
                    {label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* PREDICTOR */}
        <section className="mt-20 grid gap-8 lg:grid-cols-[1fr_0.85fr]">
          {/* INPUT */}
          <div className="rounded-[2rem] border border-[#dce2d9] bg-white p-7 shadow-sm">
            <div className="mb-6 flex items-end justify-between">
              <div>
                <h2 className="text-2xl font-semibold">
                  Analyze a protein
                </h2>

                <p className="mt-2 text-sm text-[#7a847b]">
                  Paste the primary amino-acid sequence below.
                </p>
              </div>

              <div className="flex items-center gap-4">
                {sequence && (
                  <button
                    onClick={handleClear}
                    className="text-xs font-semibold text-[#8a928b] transition hover:text-[#30422f]"
                  >
                    Clear
                  </button>
                )}

                <button
                  onClick={handleExample}
                  className="text-xs font-semibold text-[#657b62] transition hover:text-[#30422f]"
                >
                  Use example
                </button>
              </div>
            </div>

            <textarea
              value={sequence}
              onChange={(event) => {
                setSequence(event.target.value);
                setResult(null);
                setError("");
              }}
              placeholder="MALTLTPTSSVHLLSSISVARPRIFAADFNLRS..."
              className="min-h-[280px] w-full resize-none rounded-2xl border border-[#dce2d9] bg-[#fafbf9] p-5 font-mono text-sm leading-7 outline-none transition placeholder:text-[#b0b8af] focus:border-[#91a68c] focus:ring-4 focus:ring-[#e8eee4]"
              spellCheck={false}
            />

            <div className="mt-4 flex items-center justify-between text-xs text-[#899289]">
              <span>
                {cleanSequence.length} residues
              </span>

              <span>
                Amino-acid sequence
              </span>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !cleanSequence}
              className="mt-6 w-full rounded-2xl bg-[#263526] px-6 py-4 text-sm font-semibold text-white transition hover:bg-[#3b503a] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading
                ? "Analyzing sequence..."
                : "Predict enzyme activity"}
            </button>

            {error && (
              <div className="mt-4 rounded-2xl bg-[#fff2f0] px-5 py-4 text-sm text-[#a04e45]">
                {error}
              </div>
            )}
          </div>

          {/* RESULT */}
          <div className="rounded-[2rem] border border-[#dce2d9] bg-[#edf2e9] p-7">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[#687766]">
              Prediction result
            </div>

            {!result ? (
              <div className="flex min-h-[400px] items-center justify-center text-center">
                <div>
                  <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-white text-2xl shadow-sm">
                    ✦
                  </div>

                  <h3 className="text-lg font-semibold">
                    Awaiting sequence
                  </h3>

                  <p className="mt-2 max-w-xs text-sm leading-6 text-[#778176]">
                    Submit a protein sequence to start the prediction.
                  </p>
                </div>
              </div>
            ) : (
              <div className="mt-6 space-y-4">
                {/* MAIN PREDICTION */}
                <div className="rounded-3xl bg-white p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="text-xs uppercase tracking-[0.15em] text-[#899289]">
                        Predicted activity
                      </div>

                      <h3 className="mt-3 text-3xl font-semibold tracking-tight">
                        {CLASS_INFO[result.prediction].title}
                      </h3>
                    </div>

                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#e8eee4] text-[#5e7659]">
                      ✓
                    </div>
                  </div>

                  <p className="mt-4 text-sm leading-6 text-[#737d74]">
                    {CLASS_INFO[result.prediction].description}
                  </p>

                  <div className="mt-6 border-t border-[#edf0eb] pt-5">
                    <div className="flex items-end justify-between">
                      <span className="text-xs uppercase tracking-[0.15em] text-[#899289]">
                        Model confidence
                      </span>

                      <span className="text-2xl font-semibold">
                        {confidencePercent.toFixed(1)}%
                      </span>
                    </div>

                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#e8ece6]">
                      <div
                        className="h-full rounded-full bg-[#718a6c] transition-all duration-700"
                        style={{
                          width: `${Math.min(
                            confidencePercent,
                            100
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* PROBABILITIES */}
                <div className="rounded-3xl bg-white p-6">
                  <div className="text-xs uppercase tracking-[0.15em] text-[#899289]">
                    Class probabilities
                  </div>

                  <div className="mt-5 space-y-5">
                    {CLASS_ORDER.map((className) => {
                      const probability =
                        result.probabilities[className];

                      const percentage = probability * 100;
                      const isPrediction =
                        className === result.prediction;

                      return (
                        <div key={className}>
                          <div className="mb-2 flex items-center justify-between gap-4">
                            <div className="flex items-center gap-2">
                              <span
                                className={`h-2 w-2 rounded-full ${isPrediction
                                  ? "bg-[#718a6c]"
                                  : "bg-[#c8cec6]"
                                  }`}
                              />

                              <span
                                className={`text-sm ${isPrediction
                                  ? "font-semibold"
                                  : "text-[#667066]"
                                  }`}
                              >
                                {CLASS_INFO[className].shortTitle}
                              </span>
                            </div>

                            <span className="font-mono text-xs text-[#737d74]">
                              {percentage < 0.01
                                ? "<0.01"
                                : percentage.toFixed(2)}
                              %
                            </span>
                          </div>

                          <div className="h-2 overflow-hidden rounded-full bg-[#edf0eb]">
                            <div
                              className={`h-full rounded-full transition-all duration-700 ${isPrediction
                                ? "bg-[#718a6c]"
                                : "bg-[#c5cdc2]"
                                }`}
                              style={{
                                width: `${Math.max(
                                  percentage,
                                  percentage > 0 ? 0.5 : 0
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* METADATA */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="rounded-3xl bg-white p-5">
                    <div className="text-xs uppercase tracking-[0.12em] text-[#899289]">
                      Sequence
                    </div>

                    <div className="mt-2 text-2xl font-semibold">
                      {cleanSequence.length}
                    </div>

                    <div className="mt-1 text-xs text-[#899289]">
                      residues
                    </div>
                  </div>

                  <div className="rounded-3xl bg-white p-5">
                    <div className="text-xs uppercase tracking-[0.12em] text-[#899289]">
                      Model
                    </div>

                    <div className="mt-2 text-sm font-semibold">
                      ESM-2
                    </div>

                    <div className="mt-1 text-xs text-[#899289]">
                      + Logistic Regression
                    </div>
                  </div>
                </div>

                {/* DISCLAIMER */}
                <div className="rounded-2xl border border-[#dce2d9] bg-[#e6ece2] px-5 py-4 text-xs leading-5 text-[#687766]">
                  Computational prediction based on the submitted amino-acid
                  sequence. The result does not constitute experimental
                  confirmation of enzyme activity.
                </div>
              </div>
            )}
          </div>
        </section>

        {/* CLASSES */}
        <section className="mt-20">
          <div className="mb-7">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[#899289]">
              Target classes
            </div>

            <h2 className="mt-2 text-3xl font-semibold">
              Three GH13 functional groups
            </h2>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {CLASS_ORDER.map((className) => (
              <article
                key={className}
                className="rounded-3xl border border-[#dce2d9] bg-white p-6 transition hover:-translate-y-0.5 hover:shadow-[0_12px_35px_rgba(40,55,40,0.07)]"
              >
                <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-xl bg-[#edf2e9] text-sm font-semibold text-[#60745c]">
                  {CLASS_ORDER.indexOf(className) + 1}
                </div>

                <h3 className="text-lg font-semibold">
                  {CLASS_INFO[className].title}
                </h3>

                <p className="mt-3 text-sm leading-6 text-[#737d74]">
                  {CLASS_INFO[className].description}
                </p>
              </article>
            ))}
          </div>
        </section>

        {/* FOOTER */}
        <footer className="mt-20 border-t border-[#dce2d9] pt-7 text-xs text-[#899289]">
          <div className="flex flex-col justify-between gap-2 sm:flex-row">
            <span>
              Plant Enzyme ML · Research prototype
            </span>

            <span>
              GH13 sequence classification
            </span>
          </div>
        </footer>
      </section>
    </main>
  );
}
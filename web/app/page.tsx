"use client";

import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

type PredictionResponse = {
  status: string;
  sequence_length: number;
};

export default function Home() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handlePredict() {
    setError("");
    setResult(null);

    const cleanSequence = sequence
      .replace(/\s+/g, "")
      .toUpperCase();

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

  return (
    <main className="min-h-screen bg-[#f7f8f4] text-[#172018]">
      <section className="mx-auto max-w-7xl px-6 py-10 lg:px-10">
        {/* Header */}
        <header className="mb-16 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold uppercase tracking-[0.25em] text-[#657465]">
              Plant Enzyme ML
            </div>
            <div className="mt-1 text-xs text-[#899289]">
              GH13 functional specificity prediction
            </div>
          </div>

          <div className="rounded-full border border-[#d9dfd6] bg-white px-4 py-2 text-xs font-medium text-[#657465]">
            ESM-2 · ML
          </div>
        </header>

        {/* Hero */}
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
              alpha-amylase, isoamylase, or pullulanase / limit dextrinase.
            </p>
          </div>

          {/* Model card */}
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

                  <span className="text-sm font-medium">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Predictor */}
        <section className="mt-20 grid gap-8 lg:grid-cols-[1fr_0.75fr]">
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

              <button
                onClick={handleExample}
                className="text-xs font-semibold text-[#657b62] transition hover:text-[#30422f]"
              >
                Use example
              </button>
            </div>

            <textarea
              value={sequence}
              onChange={(event) => setSequence(event.target.value)}
              placeholder="MALTLTPTSSVHLLSSISVARPRIFAADFNLRS..."
              className="min-h-[280px] w-full resize-none rounded-2xl border border-[#dce2d9] bg-[#fafbf9] p-5 font-mono text-sm leading-7 outline-none transition placeholder:text-[#b0b8af] focus:border-[#91a68c] focus:ring-4 focus:ring-[#e8eee4]"
              spellCheck={false}
            />

            <div className="mt-4 flex items-center justify-between text-xs text-[#899289]">
              <span>
                {sequence.replace(/\s+/g, "").length} residues
              </span>

              <span>
                Standard amino-acid alphabet
              </span>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="mt-6 w-full rounded-2xl bg-[#263526] px-6 py-4 text-sm font-semibold text-white transition hover:bg-[#3b503a] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Analyzing sequence..." : "Predict enzyme activity"}
            </button>

            {error && (
              <div className="mt-4 rounded-2xl bg-[#fff2f0] px-5 py-4 text-sm text-[#a04e45]">
                {error}
              </div>
            )}
          </div>

          {/* Result */}
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
              <div className="mt-8">
                <div className="rounded-3xl bg-white p-6">
                  <div className="text-xs uppercase tracking-[0.15em] text-[#899289]">
                    Sequence length
                  </div>

                  <div className="mt-3 text-5xl font-semibold">
                    {result.sequence_length}
                  </div>

                  <div className="mt-2 text-sm text-[#778176]">
                    amino-acid residues
                  </div>
                </div>

                <div className="mt-4 rounded-3xl bg-white p-6">
                  <div className="text-xs uppercase tracking-[0.15em] text-[#899289]">
                    Model status
                  </div>

                  <div className="mt-3 flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-[#7c9976]" />
                    <span className="font-medium">
                      Prediction API connected
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Classes */}
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
            {[
              {
                title: "α-Amylase",
                description:
                  "Enzymes involved in hydrolysis of internal α-1,4 glycosidic bonds.",
              },
              {
                title: "Isoamylase",
                description:
                  "Debranching enzymes acting on α-1,6 linkages in starch structures.",
              },
              {
                title: "Pullulanase / LDA",
                description:
                  "Debranching activity associated with pullulan and limit dextrins.",
              },
            ].map((item) => (
              <article
                key={item.title}
                className="rounded-3xl border border-[#dce2d9] bg-white p-6"
              >
                <h3 className="text-lg font-semibold">{item.title}</h3>

                <p className="mt-3 text-sm leading-6 text-[#737d74]">
                  {item.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-20 border-t border-[#dce2d9] pt-7 text-xs text-[#899289]">
          Plant Enzyme ML · Research prototype · GH13 sequence classification
        </footer>
      </section>
    </main>
  );
}
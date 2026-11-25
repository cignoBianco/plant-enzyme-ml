"use client";

import { useState, useEffect } from "react";

// Данные из нашего итогового JSON
const DATA = {
    tissues: [
        {
            id: "tomato_root",
            species: "Томат",
            tissue: "Корень",
            icon: "🌱",
            totalCazymes: 59,
            dominantClass: "Лигнин-модифицирующие оксидоредуктазы",
            dominantCount: 34,
            benefitTitle: "Источник ферментов для делигнификации",
            benefitText: "Корень томата содержит 58% лигнин-модифицирующих ферментов (лакказы и пероксидазы). Они способны мягко разрушать лигнин — защитную оболочку биомассы. Это делает сырье доступным для дальнейшей переработки без использования агрессивной химии.",
            uses: ["Биоотбеливание бумаги", "Предобработка соломы для биогаза", "Экологичное производство"]
        },
        {
            id: "tomato_leaf",
            species: "Томат",
            tissue: "Лист",
            icon: "🍃",
            totalCazymes: 163,
            dominantClass: "Прочие гликозидгидролазы",
            dominantCount: 44,
            benefitTitle: "Источник эстераз клеточной стенки",
            benefitText: "Листья томата обогащены эстеразами (CE16), которые снимают ацетильные «замки» с гемицеллюлозы. Это повышает эффективность основных ферментов и увеличивает выход сахаров.",
            uses: ["Улучшение кормов для животных", "Производство биотоплива", "Осветление соков"]
        },
        {
            id: "potato_leaf",
            species: "Картофель",
            tissue: "Лист",
            icon: "🥔",
            totalCazymes: 244,
            dominantClass: "Прочие гликозидгидролазы",
            dominantCount: 82,
            benefitTitle: "Широкий спектр CAZyme",
            benefitText: "Лист картофеля демонстрирует самое богатое разнообразие ферментов (244 белка с обнаруженной abundance). В сочетании с ферментами томата это создает синергетический коктейль для комплексной переработки.",
            uses: ["Создание ферментных коктейлей", "Переработка картофельных очисток", "Кормовые добавки"]
        }
    ],
    mlAccuracy: 95.0
};

export default function BioconversionPage() {
    const [isProcessing, setIsProcessing] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);

    const steps = [
        "Загрузка протеомных данных...",
        "Аннотация ферментов (HMMER)...",
        "Классификация ML-моделью...",
        "Оценка биотехнологического потенциала...",
        "Анализ завершен!"
    ];

    const handleStart = () => {
        setIsProcessing(true);
        setShowResults(false);
        setCurrentStep(0);
    };

    // Логика анимации шагов
    useEffect(() => {
        if (!isProcessing) return;

        if (currentStep < steps.length - 1) {
            const timer = setTimeout(() => setCurrentStep(prev => prev + 1), 1200);
            return () => clearTimeout(timer);
        } else {
            const timer = setTimeout(() => {
                setIsProcessing(false);
                setShowResults(true);
            }, 1000);
            return () => clearTimeout(timer);
        }
    }, [isProcessing, currentStep, steps.length]);

    return (
        <main className="min-h-screen bg-[#f7f8f4] text-[#172018] font-sans pb-20">
            <section className="mx-auto max-w-6xl px-6 py-12">

                {/* Header */}
                <header className="mb-12 text-center">
                    <div className="text-sm font-semibold uppercase tracking-[0.25em] text-[#657465] mb-2">
                        Анализ биомассы
                    </div>
                    <h1 className="text-4xl lg:text-5xl font-bold tracking-tight text-[#263526]">
                        Биотехнологический потенциал <br className="hidden md:block" />
                        <span className="text-[#71856e]">пасленовых</span>
                    </h1>
                    <p className="mt-4 max-w-2xl mx-auto text-[#667066]">
                        Имитация обработки корней и листьев томата и картофеля. Система находит ферменты,
                        способные превратить сельскохозяйственные отходы в ценное сырье.
                    </p>
                </header>

                {/* Controls */}
                <div className="flex justify-center mb-12">
                    <button
                        onClick={handleStart}
                        disabled={isProcessing}
                        className="group relative inline-flex items-center gap-3 rounded-full bg-[#263526] px-8 py-4 text-sm font-semibold text-white transition hover:bg-[#3b503a] disabled:opacity-70 shadow-lg"
                    >
                        {isProcessing ? (
                            <>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Анализ...
                            </>
                        ) : (
                            <>
                                <span>🧬</span> Запустить анализ биомассы
                            </>
                        )}
                    </button>
                </div>

                {/* Processing Animation */}
                {isProcessing && (
                    <div className="mx-auto max-w-2xl bg-white rounded-3xl p-8 border border-[#dce2d9] shadow-sm mb-12 text-center">
                        <div className="mb-8 relative w-32 h-32 mx-auto">
                            <div className="absolute inset-0 rounded-full border-4 border-[#f0f4ef]"></div>
                            <div className="absolute inset-0 rounded-full border-4 border-[#718a6c] border-t-transparent animate-spin"></div>
                            <div className="absolute inset-0 flex items-center justify-center text-3xl">
                                🧪
                            </div>
                        </div>

                        <div className="space-y-4 text-left max-w-md mx-auto">
                            {steps.map((step, idx) => (
                                <div key={idx} className={`flex items-center gap-3 transition-opacity duration-300 ${idx <= currentStep ? 'opacity-100' : 'opacity-30'}`}>
                                    <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs ${idx < currentStep ? 'bg-[#718a6c] text-white' : idx === currentStep ? 'bg-[#e8eee4] text-[#718a6c]' : 'bg-gray-100 text-gray-400'}`}>
                                        {idx < currentStep ? '✓' : idx + 1}
                                    </div>
                                    <span className={`text-sm ${idx === currentStep ? 'font-semibold text-[#263526]' : 'text-[#667066]'}`}>
                                        {step}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Results */}
                {showResults && (
                    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                        {DATA.tissues.map((item, index) => (
                            <div
                                key={item.id}
                                className="bg-white rounded-3xl border border-[#dce2d9] p-6 shadow-sm hover:shadow-md transition-shadow animate-in fade-in slide-in-from-bottom-4 duration-700 ease-out fill-mode-both"
                                style={{ animationDelay: `${index * 150}ms` }}
                            >
                                {/* Header */}
                                <div className="flex items-center justify-between mb-4 border-b border-gray-100 pb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="text-3xl">{item.icon}</div>
                                        <div>
                                            <div className="text-xs font-bold text-gray-400 uppercase tracking-wider">{item.species}</div>
                                            <div className="font-semibold text-lg text-[#263526]">{item.tissue}</div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-2xl font-bold text-[#718a6c]">{item.totalCazymes}</div>
                                        <div className="text-[10px] text-gray-500 uppercase">CAZyme</div>
                                    </div>
                                </div>

                                {/* Main Stats */}
                                <div className="mb-6">
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-gray-600">Доминирующий класс:</span>
                                        <span className="font-semibold text-[#263526]">{item.dominantCount} белков</span>
                                    </div>
                                    <div className="w-full bg-gray-100 rounded-full h-2 mb-2">
                                        <div className="bg-[#718a6c] h-2 rounded-full" style={{ width: `${(item.dominantCount / item.totalCazymes) * 100}%` }}></div>
                                    </div>
                                    <p className="text-xs text-gray-500">{item.dominantClass}</p>
                                </div>

                                {/* Benefit Box */}
                                <div className="bg-[#f5f7f3] rounded-2xl p-5 mb-6">
                                    <h4 className="text-sm font-bold text-[#506150] mb-2 flex items-center gap-2">
                                        <span className="text-lg">💡</span> {item.benefitTitle}
                                    </h4>
                                    <p className="text-xs leading-relaxed text-[#667066]">
                                        {item.benefitText}
                                    </p>
                                </div>

                                {/* Uses list */}
                                <div>
                                    <h5 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">Где применить:</h5>
                                    <ul className="space-y-2">
                                        {item.uses.map((use, i) => (
                                            <li key={i} className="flex items-start text-xs text-[#3b503a]">
                                                <svg className="w-4 h-4 mr-2 text-[#718a6c] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                                </svg>
                                                {use}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Summary Block (appears after results) */}
                {showResults && (
                    <div className="mt-12 bg-[#263526] rounded-3xl p-8 text-white animate-in fade-in duration-1000 delay-500 fill-mode-both">
                        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                            <div>
                                <h3 className="text-2xl font-bold mb-2">Экономический потенциал</h3>
                                <p className="text-[#a0b39c] text-sm max-w-xl">
                                    Вместо того чтобы сжигать или выбрасывать с/х отходы, мы используем их как сырье для ферментов.
                                    Точность ML-модели, предсказавшей эти классы, составляет <strong>{DATA.mlAccuracy}%</strong>.
                                </p>
                            </div>
                            <div className="flex gap-4 shrink-0">
                                <div className="bg-[#3b503a] px-6 py-4 rounded-2xl text-center">
                                    <div className="text-3xl font-bold text-white">3</div>
                                    <div className="text-xs text-[#a0b39c] mt-1">Типа сырья</div>
                                </div>
                                <div className="bg-[#3b503a] px-6 py-4 rounded-2xl text-center">
                                    <div className="text-3xl font-bold text-white">6</div>
                                    <div className="text-xs text-[#a0b39c] mt-1">Классов ферментов</div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

            </section>
        </main>
    );
}
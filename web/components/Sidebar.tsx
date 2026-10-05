"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
    {
        href: "/dashboard",
        label: "Dashboard",
        icon: "⌂",
    },
    {
        href: "/analyze",
        label: "Analyze sequence",
        icon: "⌁",
    },
    {
        href: "/dataset",
        label: "Dataset",
        icon: "▤",
    },
    {
        href: "/proteins",
        label: "Proteins",
        icon: "◉",
    },
    {
        href: "/embeddings",
        label: "Embeddings",
        icon: "⌬",
    },
    {
        href: "/model-comparison",
        label: "Model comparison",
        icon: "◫",
    },
    {
        href: "/research",
        label: "Research",
        icon: "◎",
    },
    {
        href: "/documentation",
        label: "Documentation",
        icon: "▢",
    },
];

export default function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="hidden w-[218px] shrink-0 border-r border-[#e1e9e4] bg-white lg:flex lg:flex-col">
            <div className="flex h-[70px] items-center border-b border-[#edf1ee] px-6">
                <div className="mr-3 flex h-8 w-8 items-center justify-center rounded-xl bg-[#e5f5ed]">
                    <span className="text-lg text-[#15966e]">✦</span>
                </div>

                <div>
                    <div className="text-[13px] font-bold tracking-tight">
                        PlantEnzyme AI
                    </div>

                    <div className="text-[9px] text-[#9aa59f]">
                        research platform
                    </div>
                </div>
            </div>

            <nav className="flex-1 px-3 py-5">
                <div className="mb-3 px-3 text-[9px] font-semibold uppercase tracking-[0.16em] text-[#a1aba5]">
                    Workspace
                </div>

                {navigation.map((item) => {
                    const active =
                        pathname === item.href ||
                        pathname.startsWith(`${item.href}/`);

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`mb-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-[10px] transition ${active
                                    ? "bg-[#e9f7f1] font-semibold text-[#15966e]"
                                    : "text-[#718078] hover:bg-[#f5f8f6] hover:text-[#314239]"
                                }`}
                        >
                            <span className="flex w-4 justify-center text-[12px]">
                                {item.icon}
                            </span>

                            <span>{item.label}</span>
                        </Link>
                    );
                })}
            </nav>

            <div className="m-3 rounded-2xl bg-[#f3f8f5] p-4">
                <div className="mb-2 flex items-center gap-2">
                    <span className="text-sm text-[#15966e]">✦</span>

                    <span className="text-[10px] font-semibold text-[#52635a]">
                        Research context
                    </span>
                </div>

                <p className="text-[9px] leading-4 text-[#89968f]">
                    Computational research prototype for functional classification
                    of plant GH13 proteins.
                </p>

                <div className="mt-3 text-[8px] text-[#a0aaa5]">
                    Dataset v0.1 · ESM-2 · GH13
                </div>
            </div>
        </aside>
    );
}
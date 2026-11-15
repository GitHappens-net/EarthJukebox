import { Compass, Database, Headphones, Keyboard, Palette, type LucideIcon } from "lucide-react";

const steps: [string, string, LucideIcon][] = [
  ["Choose a dataset", "Select an image from the collection on the right.", Database],
  ["Touch the image", "Move across the large image area. Your position controls the sampled region.", Compass],
  ["Listen", "The backend sequencer sonifies the sampled colors and visual features in real time.", Headphones],
  ["Read the legend", "Use the legend below the image to understand what the colors represent.", Palette],
];

export default function InstructionsPage() {
  return (
    <div className="mx-auto max-w-7xl rounded-2xl bg-[#E5CD82]/60 p-6 shadow-md sm:p-8">
      <p className="eyebrow">Quick start</p>
      <h1 className="mb-5 mt-2 font-display text-4xl text-white text-shadow-md">How to explore</h1>
      <ol className="mt-5 grid gap-5 lg:grid-cols-2">
        {steps.map(([title, text, Icon]) => (
          <li className="flex gap-4" key={title}>
            <span>
              <strong className="flex items-center gap-2 text-2xl font-bold text-white text-shadow-md">
                <Icon size={36} aria-hidden="true" className="rounded-full bg-white p-1.5 text-[#728E6D] shadow-md" />
                {title}
              </strong>
              <span className="mt-1 block text-lg text-black/40">{text}</span>
            </span>
          </li>
        ))}
      </ol>
      <section className="mt-7 border-t border-black/10 pt-6" aria-labelledby="keyboard-shortcuts">
        <h2 id="keyboard-shortcuts" className="flex items-center gap-3 font-display text-2xl text-white text-shadow-md">
          <Keyboard size={28} className="rounded-full bg-white p-1.5 text-[#728E6D] shadow-md" aria-hidden="true" />
          Keyboard controls
        </h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[
            ["1–9", "Choose a dataset"],
            ["Space", "Mute or unmute audio"],
            ["C", "Calibrate the trackpad"],
          ].map(([key, description]) => (
            <div className="rounded-full bg-[#FFF2B4] px-3 py-2 text-sm shadow-md" key={key}>
              <kbd className="font-bold text-[#15370D]">{key}</kbd>
              <span className="ml-2 text-black/40">{description}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

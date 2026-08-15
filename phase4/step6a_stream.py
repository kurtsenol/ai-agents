import asyncio

from retail_agent import agent, build_deps
from step3_output import AnalysisResult


async def main():
    deps = build_deps()

    count = 0
    last_type = None
    repeat = 0

    async with agent.run_stream_events(
        "42 numaralı mağazada fiyat anormalliği var mı?",
        deps=deps,
        output_type=AnalysisResult,
    ) as events:

        async for event in events:
            count += 1
            current_type = type(event).__name__

            if current_type == last_type:
                repeat += 1
            else:
                if last_type is not None:
                    suffix = f" × {repeat}" if repeat > 1 else ""
                    print(f"{last_type}{suffix}")

                last_type = current_type
                repeat = 1

    # Son grubu yazdır
    if last_type is not None:
        suffix = f" × {repeat}" if repeat > 1 else ""
        print(f"{last_type}{suffix}")

    print(f"\nTotal events: {count}")


if __name__ == "__main__":
    asyncio.run(main())
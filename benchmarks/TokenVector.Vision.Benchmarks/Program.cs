namespace TokenVector.Vision.Benchmarks;

public static class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0].Equals("--gpu", StringComparison.OrdinalIgnoreCase))
        {
            GpuStressTest.Run();
        }
        else if (args.Length > 0 && args[0].Equals("--3d", StringComparison.OrdinalIgnoreCase))
        {
            Volume3DStressTest.Run();
        }
        else if (args.Length > 0 && args[0].Equals("--detection", StringComparison.OrdinalIgnoreCase))
        {
            DetectionBenchmark.Run();
        }
        else if (args.Length > 0 && args[0].Equals("--competitors", StringComparison.OrdinalIgnoreCase))
        {
            CompetitorStressTest.Run();
        }
        else if (args.Length > 0 && args[0].Equals("--standard", StringComparison.OrdinalIgnoreCase))
        {
            StressTest.Run();
        }
        else
        {
            StressTest.Run();
            Console.WriteLine();
            CompetitorStressTest.Run();
            Console.WriteLine();
            DetectionBenchmark.Run();
            Console.WriteLine();
            Volume3DStressTest.Run();
        }
    }
}



using Hockey.Shared.Dto;
using Microsoft.AspNetCore.SignalR.Client;
using Newtonsoft.Json;
using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;

namespace Hockey.Test
{
    class Program
    {
        static async Task Main(string[] args)
        {
            string address = "http://localhost:5000";

            HttpClient httpClient = new()
            {
                BaseAddress = new(address)
            };

            HubConnection connection = new HubConnectionBuilder()
                                .WithUrl($"{address}/hockeyHub")
                                .Build();

            await httpClient.PutAsJsonAsync
            (
                "hockey/video",
                new VideoInfoDto
                {
                    FileName = @"F:\VisualStudio\yolov4-deepsort-master\data\video\boba.mp4"
                }
            );

            int framesCount = -1;

            connection.On<DetectingVideoInformationDto>
            (
                "DetectionStarted",
                video =>
                {
                    Console.WriteLine($"Начало обработки видео {video.Path}");
                    Console.WriteLine($"{video.FramesCount} кадров");
                    Console.WriteLine($"Размеры: {video.Width} x {video.Height}\n");

                    framesCount = video.FramesCount;
                }
            );

            connection.On<int>
            (
                "FrameReaded",
                async frameNum =>
                {
                    if (framesCount != -1)
                    {
                        Console.WriteLine($"{100 * (frameNum / (double)framesCount):0.0} % обработано");
                    }

                    if (frameNum % 5 == 0)
                    {
                        FrameInfoDto[] frames = await httpClient.GetFromJsonAsync<FrameInfoDto[]>
                        (
                            "hockey/frames"
                        );

                        Console.WriteLine();
                        Console.WriteLine("Загрузка информации о кадрах");
                        foreach (FrameInfoDto frame in frames)
                        {
                            Console.WriteLine(JsonConvert.SerializeObject(frame, Formatting.Indented));
                        }
                        Console.WriteLine();
                    }
                }
            );
            await connection.StartAsync();
            Console.ReadLine();
        }
    }
}

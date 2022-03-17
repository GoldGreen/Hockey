using Hockey.Shared.Dto;
using ReactiveUI;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Hockey.Client.Main.Abstractions
{
    internal interface IMainModel : IReactiveObject
    {
        DetectingVideoInformationDto VideoInfo { get; set; }
        int LastProcessedFrame { get; set; }
        IDictionary<int, IReadOnlyList<PlayerDto>> FramesInfo { get; set; }

        Task StartDetectingVideo(string filePath);
        Task<IDictionary<int, IReadOnlyList<PlayerDto>>> GetAllFrames();
    }
}

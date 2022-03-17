using Hockey.Client.Main.Abstractions;
using Hockey.Client.Shared.Extensions;
using Hockey.Shared.Dto;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Threading.Tasks;
using Unity;

namespace Hockey.Client.Main.Models
{
    internal class MainModel : ReactiveObject, IMainModel
    {
        [Dependency]
        public IHockeyService HockeyService { get; init; }
        public IHockeyConnection HockeyConnection { get; }

        [Reactive] public DetectingVideoInformationDto VideoInfo { get; set; }
        [Reactive] public int LastProcessedFrame { get; set; }

        [Reactive] public IDictionary<int, IReadOnlyList<PlayerDto>> FramesInfo { get; set; }

        public MainModel(IHockeyConnection hockeyConnection)
        {
            HockeyConnection = hockeyConnection;

            HockeyConnection.FrameProcessed
                            .ObserveOnDispatcher()
                            .Subscribe(num => LastProcessedFrame = num)
                            .Cashe();

            HockeyConnection.VideoLoaded
                            .ObserveOnDispatcher()
                            .Subscribe(info => VideoInfo = info)
                            .Cashe();
        }

        public Task StartDetectingVideo(string filePath)
        {
            return HockeyService.StarDetectingtVideo(new() { FileName = filePath });
        }

        public async Task<IDictionary<int, IReadOnlyList<PlayerDto>>> GetAllFrames()
        {
            return (await HockeyService.GetAllFrames()).ToDictionary
            (
                x => x.FrameNum,
                x => x.Players as IReadOnlyList<PlayerDto>
            );
        }
    }
}

using Hockey.Shared.Dto;
using System;

namespace Hockey.Client.Main.Abstractions
{
    internal interface IHockeyConnection
    {
        IObservable<DetectingVideoInformationDto> VideoLoaded { get; }
        IObservable<int> FrameProcessed { get; }
    }
}

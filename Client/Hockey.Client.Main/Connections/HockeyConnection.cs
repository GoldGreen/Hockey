using Hockey.Client.Main.Abstractions;
using Hockey.Client.Shared.Configuration;
using Hockey.Client.Shared.Connections;
using Hockey.Client.Shared.Extensions;
using Hockey.Shared.Dto;
using Microsoft.AspNetCore.SignalR.Client;
using System;
using System.Reactive.Subjects;
using System.Threading.Tasks;

namespace Hockey.Client.Main.Connections
{
    internal class HockeyConnection : ConnectionBase, IHockeyConnection
    {
        public IObservable<int> FrameProcessed => _frameProcessed;
        public IObservable<DetectingVideoInformationDto> VideoLoaded => _videoLoaded;

        private Subject<int> _frameProcessed = new();
        private Subject<DetectingVideoInformationDto> _videoLoaded = new();

        public HockeyConnection(AddressesOption addressesOption)
            : base($"{addressesOption.Server}/hockeyHub")
        {

        }

        protected async override Task OnInit()
        {
            _connection.On<DetectingVideoInformationDto>("DetectionStarted", _videoLoaded.OnNext);
            _connection.On<int>("FrameReaded", _frameProcessed.OnNext);

            await base.OnInit();
        }
    }
}
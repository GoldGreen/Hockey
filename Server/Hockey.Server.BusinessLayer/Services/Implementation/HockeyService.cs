using Hockey.Server.BusinessLayer.Hubs;
using Hockey.Server.BusinessLayer.Services.Abstractions;
using Hockey.Shared.Dto;
using Microsoft.AspNetCore.SignalR;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Hockey.Server.BusinessLayer.Services.Implementation
{
    internal class HockeyService : IHockeyService
    {
        public IHubContext<HockeyHub> Hub { get; }
        private readonly List<FrameInfoDto> framesInfo = new();

        public HockeyService(IHubContext<HockeyHub> hub)
        {
            Hub = hub;
        }

        public async Task StartDetection(DetectingVideoInformationDto detectingVideoInformationDto)
        {
            await Hub.Clients.All.SendAsync("DetectionStarted", detectingVideoInformationDto);
        }

        public async Task AddFrameInfo(FrameInfoDto frameInfoDto)
        {
            framesInfo.Add(frameInfoDto);
            await Hub.Clients.All.SendAsync("FrameReaded", frameInfoDto.FrameNum);
        }

        public IEnumerable<FrameInfoDto> GetFrames()
        {
            return framesInfo;
        }
    }
}

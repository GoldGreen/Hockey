using Hockey.Shared.Dto;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Hockey.Server.BusinessLayer.Services.Abstractions
{
    public interface IHockeyService
    {
        IEnumerable<FrameInfoDto> GetFrames();
        Task StartDetection(DetectingVideoInformationDto detectingVideoInformationDto);
        Task AddFrameInfo(FrameInfoDto frameInfoDto);
    }
}

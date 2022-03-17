using Hockey.Shared.Dto;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Hockey.Client.Main.Abstractions
{
    internal interface IHockeyService
    {
        Task StarDetectingtVideo(VideoInfoDto videoInfoDto);
        Task<IReadOnlyList<FrameInfoDto>> GetAllFrames();
    }
}

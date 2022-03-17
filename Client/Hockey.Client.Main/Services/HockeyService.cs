using Hockey.Client.Main.Abstractions;
using Hockey.Shared.Dto;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using Unity;

namespace Hockey.Client.Main.Services
{
    internal class HockeyService : IHockeyService
    {
        [Dependency]
        public HttpClient HttpClient { get; init; }

        public async Task<IReadOnlyList<FrameInfoDto>> GetAllFrames()
        {
            return await HttpClient.GetFromJsonAsync<FrameInfoDto[]>("hockey/frames");
        }

        public Task StarDetectingtVideo(VideoInfoDto videoInfoDto)
        {
            return HttpClient.PutAsJsonAsync("hockey/video", videoInfoDto);
        }
    }
}

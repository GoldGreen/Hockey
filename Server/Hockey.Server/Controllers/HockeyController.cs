using Hockey.Server.BusinessLayer.Services.Abstractions;
using Hockey.Shared.Dto;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

namespace Hockey.Server.Controllers
{
    [ApiController]
    [Route("hockey")]
    public class HockeyController : ControllerBase
    {
        public IHockeyService HockeyService { get; }
        public IPythonInitService PythonInitService { get; }

        public HockeyController(IHockeyService hockeyService, IPythonInitService pythonInitService)
        {
            HockeyService = hockeyService;
            PythonInitService = pythonInitService;
        }

        [HttpGet("frames")]
        public IActionResult GetFrames()
        {
            return Ok(HockeyService.GetFrames());
        }

        [HttpPut("start")]
        public async Task<IActionResult> StartDetection(DetectingVideoInformationDto detectingVideoInformationDto)
        {
            await HockeyService.StartDetection(detectingVideoInformationDto);
            return Ok();
        }

        [HttpPut("frame")]
        public async Task<IActionResult> AddFrame(FrameInfoDto frameInfoDto)
        {
            await HockeyService.AddFrameInfo(frameInfoDto);
            return Ok();
        }

        [HttpPut("video")]
        public IActionResult StartVideo(VideoInfoDto videoInfoDto)
        {
            PythonInitService.Start(videoInfoDto);
            return Ok();
        }
    }
}

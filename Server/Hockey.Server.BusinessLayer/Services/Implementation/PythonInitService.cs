using Hockey.Server.BusinessLayer.Configuration;
using Hockey.Server.BusinessLayer.Services.Abstractions;
using Hockey.Shared.Dto;
using Microsoft.Extensions.Options;
using System.Diagnostics;

namespace Hockey.Server.BusinessLayer.Services.Implementation
{
    internal class PythonInitService : IPythonInitService
    {
        public PythonInitOption PythonInit { get; }

        public PythonInitService(IOptions<PythonInitOption> pythonInitOption)
        {
            PythonInit = pythonInitOption.Value;
        }

        public void Start(VideoInfoDto videoInfoDto)
        {
            ProcessStartInfo pythonInfo = new();

            pythonInfo.FileName = PythonInit.PythonPath;
            pythonInfo.Arguments = $@"{PythonInit.DetectorPath} --video {videoInfoDto.FileName}";
            pythonInfo.CreateNoWindow = false;
            pythonInfo.UseShellExecute = true;

            Process.Start(pythonInfo);
        }
    }
}

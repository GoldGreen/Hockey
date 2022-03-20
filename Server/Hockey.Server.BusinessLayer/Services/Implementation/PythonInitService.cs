using Hockey.Server.BusinessLayer.Configuration;
using Hockey.Server.BusinessLayer.Services.Abstractions;
using Hockey.Shared.Dto;
using Microsoft.Extensions.Options;
using System.Diagnostics;
using System.Linq;

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

            (string name, string value)[] args = new[]
            {
                ("video", videoInfoDto.FileName),
                ("first_team_name", videoInfoDto.FirstTeamName),
                ("second_team_name", videoInfoDto.SecondTeamName)
            };

            pythonInfo.Arguments = $@"{PythonInit.DetectorPath} {string.Join(' ', args.Where(x => !string.IsNullOrWhiteSpace(x.value))
                                                                                      .Select(x => $"--{x.name} {x.value}"))}";
            pythonInfo.CreateNoWindow = false;
            pythonInfo.UseShellExecute = true;

            Process.Start(pythonInfo);
        }
    }
}

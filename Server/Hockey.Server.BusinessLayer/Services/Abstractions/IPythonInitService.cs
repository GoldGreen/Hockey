using Hockey.Shared.Dto;

namespace Hockey.Server.BusinessLayer.Services.Abstractions
{
    public interface IPythonInitService
    {
        void Start(VideoInfoDto videoInfoDto);
    }
}
